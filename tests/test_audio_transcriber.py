"""Unit tests for the AudioTranscriber class with backend support."""

import pytest
from pathlib import Path
from src.loaders import AudioTranscriber, TranscriptionError, UnsupportedFormatError

# Check which backends are available
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from vosk import Model
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class TestAudioTranscriberBackends:
    """Test suite for AudioTranscriber backend system."""
    
    def test_auto_backend_selection(self):
        """Test that auto backend selection works."""
        if not VOSK_AVAILABLE and not WHISPER_AVAILABLE:
            with pytest.raises(TranscriptionError, match="No transcription backend available"):
                AudioTranscriber(backend="auto")
        else:
            # Should successfully create a transcriber with auto backend
            # Note: This will fail if no model is available for Vosk
            try:
                transcriber = AudioTranscriber(backend="auto")
                assert transcriber is not None
                assert transcriber.backend is not None
            except TranscriptionError as e:
                # Expected if no Vosk model is found or no backend can initialize
                error_msg = str(e).lower()
                if "model" not in error_msg and "backend available" not in error_msg:
                    raise
    
    def test_explicit_whisper_backend(self):
        """Test explicit Whisper backend selection."""
        if not WHISPER_AVAILABLE:
            with pytest.raises(TranscriptionError, match="Whisper backend not available"):
                AudioTranscriber(backend="whisper")
        else:
            transcriber = AudioTranscriber(backend="whisper", model_name="base")
            assert transcriber is not None
            assert transcriber.backend_name == "whisper"
    
    def test_explicit_vosk_backend(self):
        """Test explicit Vosk backend selection."""
        if not VOSK_AVAILABLE:
            with pytest.raises(TranscriptionError, match="Vosk backend not available"):
                AudioTranscriber(backend="vosk")
        else:
            # Try to create with default model path
            # This will fail if no model is available
            try:
                transcriber = AudioTranscriber(backend="vosk")
                # If successful, verify it's a Vosk backend
                assert transcriber.backend_name == "vosk"
            except TranscriptionError as e:
                # Expected if no model is found
                assert "model" in str(e).lower()
    
    def test_invalid_backend_name(self):
        """Test that invalid backend name raises error."""
        with pytest.raises(TranscriptionError, match="Unknown backend"):
            AudioTranscriber(backend="invalid-backend")
    
    @pytest.mark.skipif(not VOSK_AVAILABLE, reason="Vosk not available")
    def test_vosk_supported_formats(self):
        """Test that Vosk backend has correct supported formats."""
        try:
            transcriber = AudioTranscriber(backend="vosk")
            assert '.wav' in transcriber.SUPPORTED_FORMATS
        except TranscriptionError:
            # Expected if no model available
            pytest.skip("Vosk model not available")
    
    @pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
    def test_whisper_supported_formats(self):
        """Test that Whisper backend has correct supported formats."""
        transcriber = AudioTranscriber(backend="whisper", model_name="base")
        expected_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
        assert transcriber.SUPPORTED_FORMATS == expected_formats


class TestAudioTranscriberErrorHandling:
    """Test suite for AudioTranscriber error handling."""
    
    @pytest.fixture
    def transcriber(self):
        """Create an AudioTranscriber instance for testing."""
        try:
            return AudioTranscriber(backend="auto")
        except TranscriptionError:
            pytest.skip("No transcription backend available")
    
    def test_file_not_found_error(self, transcriber):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe("nonexistent_audio.wav")
    
    def test_file_not_found_error_with_timestamps(self, transcriber):
        """Test that FileNotFoundError is raised for non-existent files with timestamps."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcriber.transcribe_with_timestamps("nonexistent_audio.wav")
    
    def test_unsupported_format_error(self, transcriber, tmp_path):
        """Test that UnsupportedFormatError is raised for unsupported formats."""
        # Create a temporary file with unsupported extension
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("This is not an audio file")
        
        with pytest.raises(UnsupportedFormatError, match="Unsupported audio format"):
            transcriber.transcribe(str(unsupported_file))
