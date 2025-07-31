"""Custom exceptions for Voxtral package."""


class VoxtralError(Exception):
    """Base exception for all Voxtral-related errors."""
    
    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class VoxtralServerError(VoxtralError):
    """Exception raised when the vLLM server encounters an error."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int | None = None, 
        details: str | None = None
    ) -> None:
        super().__init__(message, details)
        self.status_code = status_code


class VoxtralConfigError(VoxtralError):
    """Exception raised for configuration-related errors."""
    pass


class VoxtralAudioError(VoxtralError):
    """Exception raised for audio processing errors."""
    pass


class VoxtralTimeoutError(VoxtralError):
    """Exception raised when operations timeout."""
    pass
