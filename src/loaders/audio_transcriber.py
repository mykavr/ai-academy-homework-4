"""Audio transcriber module with pluggable backend support."""

from typing import List, Dict, Optional
from .transcription_backends import (
    TranscriptionBackend,
    TranscriptionError,
    UnsupportedFormatError,
    WhisperBackend,
    VoskBackend,
    WHISPER_AVAILABLE,
    VOSK_AVAILABLE
)


class AudioTranscriber:
    """
    Transcribes audio files to text using pluggable backends.
    
    Supports multiple transcription engines:
    - Whisper (OpenAI): High accuracy, multiple formats, requires Python < 3.14
    - Vosk: Offline, fast, works with Python 3.14, requires WAV format
    """
    
    def __init__(self, backend: str = "auto", **backend_kwargs):
        """
        Initialize the AudioTranscriber with a specific backend.
        
        Args:
            backend: Backend to use for transcription.
                    Options: "whisper", "vosk", "auto"
                    - "auto": Automatically select best available backend
                    - "whisper": Use OpenAI Whisper (requires openai-whisper)
                    - "vosk": Use Vosk (requires vosk)
            **backend_kwargs: Additional arguments passed to the backend constructor
                    For Whisper: model_name="base" (tiny, base, small, medium, large)
                    For Vosk: model_path="/path/to/vosk/model"
                    
        Raises:
            TranscriptionError: If the backend fails to initialize
        """
        self.backend_name = backend
        self.backend = self._create_backend(backend, **backend_kwargs)
    
    def _create_backend(self, backend: str, **kwargs) -> TranscriptionBackend:
        """Create and return the appropriate backend."""
        if backend == "auto":
            # Try backends in order of preference
            if VOSK_AVAILABLE:
                try:
                    return VoskBackend(**kwargs)
                except TranscriptionError:
                    pass  # Try next backend
            
            if WHISPER_AVAILABLE:
                try:
                    return WhisperBackend(**kwargs)
                except TranscriptionError:
                    pass
            
            # No backend available
            raise TranscriptionError(
                "No transcription backend available. Please install one of:\n"
                "  - vosk: pip install vosk (recommended for Python 3.14)\n"
                "  - openai-whisper: pip install openai-whisper (requires Python < 3.14)"
            )
        
        elif backend == "whisper":
            if not WHISPER_AVAILABLE:
                raise TranscriptionError(
                    "Whisper backend not available. Install with: pip install openai-whisper"
                )
            return WhisperBackend(**kwargs)
        
        elif backend == "vosk":
            if not VOSK_AVAILABLE:
                raise TranscriptionError(
                    "Vosk backend not available. Install with: pip install vosk"
                )
            return VoskBackend(**kwargs)
        
        else:
            raise TranscriptionError(
                f"Unknown backend: {backend}. Options: 'whisper', 'vosk', 'auto'"
            )
    
    @property
    def SUPPORTED_FORMATS(self):
        """Get supported formats for the current backend."""
        return self.backend.SUPPORTED_FORMATS
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text as a single string
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            UnsupportedFormatError: If the audio format is not supported
            TranscriptionError: If transcription fails
        """
        return self.backend.transcribe(audio_path)
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Transcribe an audio file with timestamps for each segment.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of dictionaries, each containing:
                - 'start': Start time in seconds
                - 'end': End time in seconds
                - 'text': Transcribed text for this segment
                
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            UnsupportedFormatError: If the audio format is not supported
            TranscriptionError: If transcription fails
        """
        return self.backend.transcribe_with_timestamps(audio_path)
