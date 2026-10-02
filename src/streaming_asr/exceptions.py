class ConfigurationError(ValueError):
    """Raised when a configuration is invalid or incomplete."""


class BackendNotAvailableError(RuntimeError):
    """Raised when a requested optional backend cannot be used."""


class InvalidAudioChunkError(ValueError):
    """Raised when an audio chunk violates the audio contract."""


class StreamingStateError(RuntimeError):
    """Raised when a streaming lifecycle operation is invalid."""

