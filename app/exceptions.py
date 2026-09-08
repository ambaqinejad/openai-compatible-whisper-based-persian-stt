# app/exceptions.py


class STTException(Exception):
    """Base exception for the application."""


class AudioProcessingError(STTException):
    """Audio decoding/conversion failed."""


class AudioTooLargeError(STTException):
    """Uploaded audio is too large."""


class InvalidAudioError(STTException):
    """Uploaded file is not a valid audio/media file."""


class TranscriptionError(STTException):
    """Model transcription failed."""


class ModelLoadError(STTException):
    """Model failed to load."""


class AuthenticationError(STTException):
    """Invalid API key."""