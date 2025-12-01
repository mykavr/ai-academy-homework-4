"""Tests for VideoProcessor class."""

import pytest
from pathlib import Path
import tempfile
import os

from src.loaders.video_processor import (
    VideoProcessor,
    VideoProcessingError,
    UnsupportedFormatError
)


class TestVideoProcessor:
    """Test suite for VideoProcessor."""
    
    def test_initialization(self):
        """Test that VideoProcessor initializes correctly."""
        processor = VideoProcessor()
        assert processor is not None
        assert processor.transcriber is not None
    
    def test_unsupported_format_error(self):
        """Test that unsupported video formats raise appropriate error."""
        processor = VideoProcessor()
        
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(UnsupportedFormatError) as exc_info:
                processor.process_video(temp_path)
            
            assert "Unsupported video format" in str(exc_info.value)
            assert ".txt" in str(exc_info.value)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_file_not_found_error(self):
        """Test that missing video files raise FileNotFoundError."""
        processor = VideoProcessor()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            processor.process_video("nonexistent_video.mp4")
        
        assert "Video file not found" in str(exc_info.value)
    
    def test_supported_formats_defined(self):
        """Test that supported formats are properly defined."""
        assert hasattr(VideoProcessor, 'SUPPORTED_FORMATS')
        assert '.mp4' in VideoProcessor.SUPPORTED_FORMATS
        assert '.avi' in VideoProcessor.SUPPORTED_FORMATS
        assert '.mov' in VideoProcessor.SUPPORTED_FORMATS
        assert '.mkv' in VideoProcessor.SUPPORTED_FORMATS
    
    def test_extract_audio_file_not_found(self):
        """Test extract_audio with non-existent file."""
        processor = VideoProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.extract_audio("nonexistent.mp4", "output.wav")
    
    def test_extract_audio_unsupported_format(self):
        """Test extract_audio with unsupported format."""
        processor = VideoProcessor()
        
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(UnsupportedFormatError):
                processor.extract_audio(temp_path, "output.wav")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
