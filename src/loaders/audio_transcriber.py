"""Audio transcriber using Vosk with automatic format conversion."""

from pathlib import Path
from typing import List, Dict
import wave
import json
import tempfile
import os

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    Model = None
    KaldiRecognizer = None

import subprocess
import shutil

# Check if ffmpeg is available
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None


class TranscriptionError(Exception):
    """Exception raised when audio transcription fails."""
    pass


class UnsupportedFormatError(Exception):
    """Exception raised when audio format is not supported."""
    pass


class AudioTranscriber:
    """
    Transcribes audio files to text using Vosk.
    
    Automatically converts audio files to the required format (mono WAV, 16kHz).
    Supports: MP3, M4A, AAC, WAV, AIFF, FLAC, OGG, and more.
    """
    
    # Supported input formats (will be converted to WAV if needed)
    SUPPORTED_FORMATS = {
        '.mp3', '.m4a', '.aac', '.wav', '.aiff', '.aif',
        '.flac', '.ogg', '.opus', '.webm', '.wma'
    }
    
    def __init__(self, model_path: str = None):
        """
        Initialize the AudioTranscriber with a Vosk model.
        
        Args:
            model_path: Path to the Vosk model directory (uses config default if not specified)
                       
        Raises:
            TranscriptionError: If Vosk or the model fails to load
        """
        # Import here to avoid circular dependency
        if model_path is None:
            from ..config import default_config
            model_path = default_config.vosk_model_path
        if not VOSK_AVAILABLE:
            raise TranscriptionError(
                "Vosk library is not available. Please install vosk: pip install vosk"
            )
        
        if not FFMPEG_AVAILABLE:
            raise TranscriptionError(
                "ffmpeg is not available. Please install ffmpeg:\n"
                "  Windows: Download from https://ffmpeg.org/download.html\n"
                "  Mac: brew install ffmpeg\n"
                "  Linux: sudo apt-get install ffmpeg"
            )
        
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Vosk model."""
        try:
            if not Path(self.model_path).exists():
                raise TranscriptionError(
                    f"Vosk model not found at: {self.model_path}\n"
                    "Download a model using: python download_vosk_model.py\n"
                    "Or from: https://alphacephei.com/vosk/models"
                )
            self.model = Model(self.model_path)
        except Exception as e:
            raise TranscriptionError(
                f"Failed to load Vosk model from '{self.model_path}': {str(e)}"
            )
    
    def _convert_to_wav(self, audio_path: str) -> str:
        """
        Convert audio file to mono WAV format (16kHz) required by Vosk.
        
        Uses ffmpeg to convert any audio format to the required format.
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Path to the converted WAV file (temporary file)
            
        Raises:
            TranscriptionError: If conversion fails
        """
        try:
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Use ffmpeg to convert to mono WAV at 16kHz
            # -i: input file
            # -ar 16000: set sample rate to 16kHz
            # -ac 1: set to mono (1 channel)
            # -y: overwrite output file
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-y',
                temp_wav.name
            ]
            
            # Run ffmpeg (suppress output)
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return temp_wav.name
            
        except subprocess.CalledProcessError as e:
            # Clean up temp file on error
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
            raise TranscriptionError(
                f"Failed to convert audio file: {audio_path}\n"
                f"ffmpeg error: {e.stderr.decode('utf-8', errors='ignore')}"
            )
        except Exception as e:
            # Clean up temp file on error
            if 'temp_wav' in locals() and os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
            raise TranscriptionError(
                f"Failed to convert audio file: {audio_path}. Error: {str(e)}"
            )
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        
        Automatically converts the audio to the required format if needed.
        
        Args:
            audio_path: Path to the audio file (any supported format)
            
        Returns:
            Transcribed text as a single string
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            UnsupportedFormatError: If the audio format is not supported
            TranscriptionError: If transcription fails
        """
        path = Path(audio_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check if format is supported
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        temp_wav_path = None
        try:
            # Convert to WAV if needed
            if path.suffix.lower() != '.wav':
                temp_wav_path = self._convert_to_wav(str(path))
                wav_path = temp_wav_path
            else:
                # Check if WAV is already in correct format
                with wave.open(str(path), "rb") as wf:
                    if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                        # Need to convert
                        temp_wav_path = self._convert_to_wav(str(path))
                        wav_path = temp_wav_path
                    else:
                        wav_path = str(path)
            
            # Transcribe the WAV file
            wf = wave.open(wav_path, "rb")
            
            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(False)
            
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
            
        except (TranscriptionError, FileNotFoundError, UnsupportedFormatError):
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Failed to transcribe audio file: {audio_path}. Error: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass  # Ignore cleanup errors
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict]:
        """
        Transcribe an audio file with timestamps for each segment.
        
        Automatically converts the audio to the required format if needed.
        
        Args:
            audio_path: Path to the audio file (any supported format)
            
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
        path = Path(audio_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check if format is supported
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        temp_wav_path = None
        try:
            # Convert to WAV if needed
            if path.suffix.lower() != '.wav':
                temp_wav_path = self._convert_to_wav(str(path))
                wav_path = temp_wav_path
            else:
                # Check if WAV is already in correct format
                with wave.open(str(path), "rb") as wf:
                    if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                        # Need to convert
                        temp_wav_path = self._convert_to_wav(str(path))
                        wav_path = temp_wav_path
                    else:
                        wav_path = str(path)
            
            # Transcribe the WAV file
            wf = wave.open(wav_path, "rb")
            
            # Create recognizer with word timestamps
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            # Process audio
            segments = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if 'result' in result and result['result']:
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
            
        except (TranscriptionError, FileNotFoundError, UnsupportedFormatError):
            raise
        except Exception as e:
            raise TranscriptionError(
                f"Failed to transcribe audio file: {audio_path}. Error: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass  # Ignore cleanup errors
