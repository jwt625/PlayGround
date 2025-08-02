"""Exception handling for the FastAPI application."""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from voxtral.exceptions import (
    VoxtralError,
    VoxtralServerError,
    VoxtralAudioError,
    VoxtralTimeoutError,
)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": str(exc),
                "errors": exc.errors(),
            }
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
            }
        },
    )


async def voxtral_exception_handler(request: Request, exc: VoxtralError) -> JSONResponse:
    """Handle Voxtral-specific exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "SERVER_ERROR"
    
    if isinstance(exc, VoxtralServerError):
        status_code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        error_code = "MODEL_UNAVAILABLE"
    elif isinstance(exc, VoxtralAudioError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = "AUDIO_PROCESSING_ERROR"
    elif isinstance(exc, VoxtralTimeoutError):
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
        error_code = "PROCESSING_TIMEOUT"
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": str(exc),
                "details": exc.details if hasattr(exc, "details") else None,
            }
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc),
            }
        },
    )


def register_exception_handlers(app) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(VoxtralError, voxtral_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
