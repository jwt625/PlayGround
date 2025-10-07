"""FastAPI server for GLM-4.5-Air with authentication and optimal H100 configuration."""

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .config import InferenceConfig, get_optimized_h100_config
from .vllm_server import GLMVLLMServer

# Global server instance
glm_server: GLMVLLMServer | None = None
config: InferenceConfig | None = None

# Security
security = HTTPBearer()

logger = logging.getLogger(__name__)


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(default=2048, ge=1, le=8192, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=-1, description="Top-k sampling parameter (-1 to disable)")
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream the response")


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int] | None = None
    generation_time: float | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model: str
    tensor_parallel_size: int
    startup_time: float | None
    gpu_memory_usage: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str | None = None


# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API key authentication."""
    if not config or not config.require_auth:
        return "no-auth-required"

    if credentials.credentials != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global glm_server, config

    # Startup
    logger.info("Starting GLM-4.5-Air server...")

    # Get optimized configuration
    config = get_optimized_h100_config()
    logger.info(f"Generated API key: {config.api_key}")
    logger.info(f"Server will run on {config.host}:{config.port}")

    # Initialize GLM server
    glm_server = GLMVLLMServer(config)
    await glm_server.initialize()

    logger.info("GLM-4.5-Air server started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down GLM-4.5-Air server...")
    if glm_server:
        await glm_server.shutdown()
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="GLM-4.5-Air API Server",
    description="Production-ready GLM-4.5-Air inference server with authentication",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not glm_server:
        raise HTTPException(status_code=503, detail="Server not initialized")

    health_data = await glm_server.health_check()

    return HealthResponse(
        status=health_data["status"],
        model=health_data.get("model", "unknown"),
        tensor_parallel_size=health_data.get("tensor_parallel_size", 0),
        startup_time=health_data.get("startup_time"),
        gpu_memory_usage=health_data.get("gpu_memory_usage")
    )


@app.post("/v1/completions", response_model=GenerationResponse)
async def create_completion(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a text completion (OpenAI-compatible endpoint)."""
    if not glm_server:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming not supported in this endpoint. Use /v1/completions/stream"
            )

        result = await glm_server.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else None,
            stop=request.stop,
            stream=False
        )

        return GenerationResponse(**result)

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions/stream")
async def create_completion_stream(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a streaming text completion."""
    if not glm_server:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        async def generate_stream():
            async for chunk in await glm_server.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else None,
                stop=request.stop,
                stream=True
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Generate text (custom endpoint)."""
    return await create_completion(request, api_key)


@app.get("/config")
async def get_server_config(api_key: str = Depends(verify_api_key)):
    """Get server configuration (admin endpoint)."""
    if not config:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Return config without sensitive information
    config_dict = config.model_dump()
    config_dict.pop("api_key", None)  # Remove API key from response

    return config_dict


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: str = "models/GLM-4.5-Air-FP8",
    log_level: str = "info"
):
    """Run the GLM-4.5-Air server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run server
    uvicorn.run(
        app,  # Use the app directly instead of string import
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=True,
        reload=False,  # Disable reload for production
        workers=1,  # Single worker for GPU models
    )


if __name__ == "__main__":
    run_server()
