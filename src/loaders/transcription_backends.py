"""Transcription backend implementations for different speech-to-text engines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
import warnings


class TranscriptionError(Exception):
    """Exception raised when audio transcription fails."""
    pass


class UnsupportedFormatError(Exception):
    """Exception raised when audio format is not supported."""
    pass


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""
    
    SUPPORTED_FORMATS = set()
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def _validate_file(self, audio_path: str) -> Path:
        """Validate that the audio file exists and has a supported format."""
        path = Path(audio_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        return path


# Whisper Backend
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None


class WhisperBackend(TranscriptionBackend):
    """Whisper-based transcription backend."""
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the Whisper backend.
        
        Args:
            model_name: Name of the Whisper model to use.
                       Options: tiny, base, small, medium, large
                       Default: base (good balance of speed and accuracy)
                       
        Raises:
            TranscriptionError: If the model fails to load
        """
        if not WHISPER_AVAILABLE:
            raise TranscriptionError(
                "Whisper library is not available. Please install openai-whisper: "
                "pip install openai-whisper"
            )
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
            self.model = whisper.load_model(self.model_name)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load Whisper model '{self.model_name}': {str(e)}"
            )
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text using Whisper."""
        path = self._validate_file(audio_path)
        
        try:
            result = self.model.transcribe(str(path))
            return result.get('text', '').strip()
        except Exception as e:
            raise TranscriptionError(
                f"Failed to transcribe audio file: {audio_path}. Error: {str(e)}"
            )
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """Transcribe an audio file with timestamps using Whisper."""
        path = self._validate_file(audio_path)
        
        try:
            result = self.model.transcribe(str(path))
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': segment.get('start', 0.0),
                    'end': segment.get('end', 0.0),
                    'text': segment.get('text', '').strip()
                })
            return segments
        except Exception as e:
            raise TranscriptionError(
                f"Failed to transcribe audio file: {audio_path}. Error: {str(e)}"
            )


# Vosk Backend
try:
    from vosk import Model, KaldiRecognizer
    import wave
    import json
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    Model = None
    KaldiRecognizer = None


class VoskBackend(TranscriptionBackend):
    """Vosk-based transcription backend."""
    
    SUPPORTED_FORMATS = {'.wav'}  # Vosk primarily works with WAV files
    
    def __init__(self, model_path: str = None):
        """
        Initialize the Vosk backend.
        
        Args:
            model_path: Path to the Vosk model directory.
                       If None, will try to use a default model.
                       Download models from: https://alphacephei.com/vosk/models
                       
        Raises:
            TranscriptionError: If the model fails to load
        """
        if not VOSK_AVAILABLE:
            raise TranscriptionError(
                "Vosk library is not available. Please install vosk: "
                "pip install vosk"
            )
        
        self.model_path = model_path or self._get_default_model_path()
        self.model = None
        self._load_model()
    
    def _get_default_model_path(self) -> str:
        """Get the default model path."""
        # Try common locations
        default_paths = [
            "./models/vosk-model-small-en-us-0.15",
            "./vosk-model",
            str(Path.home() / "vosk-model"),
        ]
        
        for path in default_paths:
            if Path(path).exists():
                return path
        
        raise TranscriptionError(
            "No Vosk model found. Please download a model from "
            "https://alphacephei.com/vosk/models and specify the path, "
            "or place it in one of these locations: " + ", ".join(default_paths)
        )
    
    def _load_model(self):
        """Load the Vosk model."""
        try:
            if not Path(self.model_path).exists():
                raise TranscriptionError(
                    f"Vosk model not found at: {self.model_path}. "
                    "Download models from: https://alphacephei.com/vosk/models"
                )
            self.model = Model(self.model_path)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load Vosk model from '{self.model_path}': {str(e)}"
            )
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text using Vosk."""
        path = self._validate_file(audio_path)
        
        try:
            # Open the WAV file
            wf = wave.open(str(path), "rb")
            
            # Check if the audio is in the correct format
            if wf.getnchannels() != 1:
                wf.close()
                raise TranscriptionError(
                    f"Audio must be mono (1 channel). Found {wf.getnchannels()} channels. "
                    "Please convert to mono first."
                )
            
            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(False)  # Don't need word-level timestamps for basic transcription
            
            # Process audio
            full_text = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if 'text' in result and result['text']:
                        full_text.append(result['text'])
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if 'text' in final_result and final_result['text']:
                full_text.append(final_result['text'])
            
            wf.close()
            
            return ' '.join(full_text).strip()
            
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Failed to transcribe audio file: {audio_path}. Error: {str(e)}"
            )
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """Transcribe an audio file with timestamps using Vosk."""
        path = self._validate_file(audio_path)
        
        try:
            # Open the WAV file
            wf = wave.open(str(path), "rb")
            
            # Check if the audio is in the correct format
            if wf.getnchannels() != 1:
                wf.close()
                raise TranscriptionError(
                    f"Audio must be mono (1 channel). Found {wf.getnchannels()} channels. "
                    "Please convert to mono first."
                )
            
            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)  # Enable word-level timestamps
            
            # Process audio
            segments = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if 'result' in result and result['result']:
                        # Group words into segments
                        words = result['result']
                        if words:
                            segment_text = ' '.join([w['word'] for w in words])
                            segments.append({
                                'start': words[0]['start'],
                                'end': words[-1]['end'],
                                'text': segment_text
                            })
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if 'result' in final_result and final_result['result']:
                words = final_result['result']
                if words:
                    segment_text = ' '.join([w['word'] for w in words])
                    segments.append({
                        'start': words[0]['start'],
                        'end': words[-1]['end'],
                        'text': segment_text
                    })
            
            wf.close()
            
            return segments
            
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Failed to transcribe audio file: {audio_path}. Error: {str(e)}"
            )
