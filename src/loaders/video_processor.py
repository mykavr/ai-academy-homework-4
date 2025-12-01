"""Video processor for extracting audio and transcribing video files."""

from pathlib import Path
from typing import Optional
import tempfile
import os

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None

from .audio_transcriber import AudioTranscriber, TranscriptionError


class UnsupportedFormatError(Exception):
    """Exception raised when video format is not supported."""
    pass


class VideoProcessingError(Exception):
    """Exception raised when video processing fails."""
    pass


class VideoProcessor:
    """
    Processes video files by extracting audio and transcribing.
    
    Supports common video formats: MP4, AVI, MOV, MKV, WEBM, FLV, WMV.
    """
    
    # Supported video formats
    SUPPORTED_FORMATS = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'
    }
    
    def __init__(self, model_path: str = "models/vosk-model-small-en-us-0.15"):
        """
        Initialize the VideoProcessor with an audio transcriber.
        
        Args:
            model_path: Path to the Vosk model directory for transcription
                       
        Raises:
            VideoProcessingError: If moviepy is not available or transcriber fails to initialize
        """
        if not MOVIEPY_AVAILABLE:
            raise VideoProcessingError(
                "moviepy library is not available. Please install moviepy: pip install moviepy"
            )
        
        try:
            self.transcriber = AudioTranscriber(model_path=model_path)
        except Exception as e:
            raise VideoProcessingError(
                f"Failed to initialize audio transcriber: {str(e)}"
            )
    
    def extract_audio(self, video_path: str, output_path: str) -> str:
        """
        Extract audio track from a video file.
        
        Args:
            video_path: Path to the input video file
            output_path: Path where the extracted audio should be saved
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
            UnsupportedFormatError: If the video format is not supported
            VideoProcessingError: If audio extraction fails
        """
        path = Path(video_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if format is supported
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        try:
            # Load video file
            video = VideoFileClip(str(path))
            
            # Check if video has audio
            if video.audio is None:
                video.close()
                raise VideoProcessingError(
                    f"Video file has no audio track: {video_path}"
                )
            
            # Extract audio
            video.audio.write_audiofile(
                output_path,
                codec='pcm_s16le',  # WAV format
                fps=16000,  # 16kHz sample rate for Vosk
                nbytes=2,
                buffersize=2000,
                logger=None  # Suppress moviepy output
            )
            
            # Close video to free resources
            video.close()
            
            return output_path
            
        except (FileNotFoundError, UnsupportedFormatError, VideoProcessingError):
            raise
        except Exception as e:
            raise VideoProcessingError(
                f"Failed to extract audio from video: {video_path}. Error: {str(e)}"
            )
    
    def process_video(self, video_path: str) -> str:
        """
        Process a video file by extracting audio and transcribing it.
        
        This method automatically:
        1. Extracts the audio track from the video
        2. Transcribes the audio to text
        3. Cleans up temporary files
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Transcribed text from the video's audio track
            
        Raises:
            FileNotFoundError: If the video file doesn't exist
            UnsupportedFormatError: If the video format is not supported
            VideoProcessingError: If video processing or transcription fails
        """
        path = Path(video_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if format is supported
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        temp_audio_path = None
        try:
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            temp_audio_path = temp_audio.name
            
            # Extract audio from video
            self.extract_audio(str(path), temp_audio_path)
            
            # Transcribe the extracted audio
            transcription = self.transcriber.transcribe(temp_audio_path)
            
            return transcription
            
        except (FileNotFoundError, UnsupportedFormatError, VideoProcessingError):
            raise
        except TranscriptionError as e:
            raise VideoProcessingError(
                f"Failed to transcribe audio from video: {video_path}. Error: {str(e)}"
            )
        except Exception as e:
            raise VideoProcessingError(
                f"Failed to process video: {video_path}. Error: {str(e)}"
            )
        finally:
            # Clean up temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass  # Ignore cleanup errors
