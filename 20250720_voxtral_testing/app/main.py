"""Main FastAPI application for Voxtral transcription service."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .core.config import config
from .core.exceptions import register_exception_handlers
from .core.logging import setup_logging, LoggingMiddleware, get_logger
from .routers import health, transcription, understanding, streaming
from .services.voxtral_service import voxtral_service

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {config.app_name} v{config.app_version}")
    
    # Check Voxtral backend health
    try:
        is_healthy = await voxtral_service.health_check()
        if is_healthy:
            model_name = await voxtral_service.get_model_name()
            logger.info(f"Connected to Voxtral backend: {model_name}")
        else:
            logger.warning("Voxtral backend is not available at startup")
    except Exception as e:
        logger.error(f"Failed to connect to Voxtral backend: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="""
    **Voxtral Transcription API** provides real-time audio transcription and understanding capabilities
    using Mistral AI's Voxtral Mini 3B model.
    
    ## Features
    
    - **Audio Transcription**: Convert speech to text with high accuracy
    - **Audio Understanding**: Ask questions about audio content
    - **Batch Processing**: Process multiple audio files efficiently
    - **Streaming Support**: Real-time transcription via WebSocket
    - **Multiple Formats**: Support for MP3, WAV, FLAC, M4A, OGG
    - **Multilingual**: Support for 8 languages (EN, ES, FR, PT, HI, DE, NL, IT)
    
    ## Rate Limits
    
    - **Transcription**: 50 requests/minute
    - **Understanding**: 20 requests/minute
    - **Batch**: 10 requests/minute
    - **General**: 100 requests/minute
    
    ## Audio Constraints
    
    - **Max file size**: 100MB
    - **Max duration**: 30 minutes (transcription), 40 minutes (understanding)
    - **Supported formats**: MP3, WAV, FLAC, M4A, OGG
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=config.allowed_methods,
    allow_headers=config.allowed_headers,
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(health.router)
app.include_router(transcription.router)
app.include_router(understanding.router)
app.include_router(streaming.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.app_name,
        "version": config.app_version,
        "description": "Voxtral transcription and audio understanding API",
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "transcription": "/transcribe",
            "batch_transcription": "/transcribe/batch",
            "audio_understanding": "/understand",
            "streaming_transcription": "/stream/transcribe",
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info",
    )
