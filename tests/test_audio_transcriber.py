"""Unit tests for the AudioTranscriber class."""

import pytest
from pathlib import Path
from src.loaders import AudioTranscriber, TranscriptionError, UnsupportedFormatError

# Check if vosk is available
try:
    from vosk import Model
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class TestAudioTranscriber:
    """Test suite for AudioTranscriber."""
    
    @pytest.fixture
    def transcriber(self):
        """Create an AudioTranscriber instance for testing."""
        if not VOSK_AVAILABLE:
            pytest.skip("Vosk not available")
        
        try:
            return AudioTranscriber(model_path="models/vosk-model-small-en-us-0.15")
        except TranscriptionError:
            pytest.skip("Vosk model not available")
    
    def test_transcriber_initialization(self, transcriber):
        """Test that the transcriber initializes correctly."""
        assert transcriber is not None
        assert transcriber.model is not None
        assert transcriber.model_path == "models/vosk-model-small-en-us-0.15"
    
    def test_supported_formats(self, transcriber):
        """Test that supported formats are correctly defined."""
        expected_formats = {
            '.mp3', '.m4a', '.aac', '.wav', '.aiff', '.aif',
            '.flac', '.ogg', '.opus', '.webm', '.wma'
        }
        assert transcriber.SUPPORTED_FORMATS == expected_formats
    
    def test_file_not_found_error(self, transcriber):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe("nonexistent_audio.mp3")
    
    def test_file_not_found_error_with_timestamps(self, transcriber):
        """Test that FileNotFoundError is raised for non-existent files with timestamps."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe_with_timestamps("nonexistent_audio.mp3")
    
    def test_unsupported_format_error(self, transcriber, tmp_path):
        """Test that UnsupportedFormatError is raised for unsupported formats."""
        # Create a temporary file with unsupported extension
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("This is not an audio file")
        
        with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
            transcriber.transcribe(str(unsupported_file))
    
    def test_unsupported_format_error_with_timestamps(self, transcriber, tmp_path):
        """Test that UnsupportedFormatError is raised for unsupported formats with timestamps."""
        # Create a temporary file with unsupported extension
        unsupported_file = tmp_path / "test.doc"
        unsupported_file.write_text("This is not an audio file")
        
        with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
            transcriber.transcribe_with_timestamps(str(unsupported_file))
    
    def test_vosk_not_available_raises_error(self):
        """Test that TranscriptionError is raised when vosk is not available."""
        if VOSK_AVAILABLE:
            pytest.skip("Vosk is available, cannot test unavailable case")
        
        with pytest.raises(TranscriptionError, match="Vosk library is not available"):
            AudioTranscriber()
    
    def test_invalid_model_path_raises_error(self):
        """Test that TranscriptionError is raised for invalid model path."""
        if not VOSK_AVAILABLE:
            pytest.skip("Vosk not available")
        
        with pytest.raises(TranscriptionError, match="model not found"):
            AudioTranscriber(model_path="invalid/model/path")
