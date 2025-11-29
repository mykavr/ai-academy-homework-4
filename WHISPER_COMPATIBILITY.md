# Audio Transcription - Multi-Backend Support

## Current Status

The `AudioTranscriber` class now supports **multiple transcription backends** with automatic selection:
- ✅ **Vosk**: Works with Python 3.14, offline, fast (recommended)
- ✅ **Whisper**: High accuracy, multiple formats (requires Python < 3.14)

## Backend System

The AudioTranscriber uses a pluggable backend architecture that allows easy switching between different speech-to-text engines.

### Vosk Backend (Recommended for Python 3.14)

**Advantages:**
- ✅ Works with Python 3.14
- ✅ Fully offline (no internet required)
- ✅ Fast transcription
- ✅ No dependency issues

**Requirements:**
- Install: `pip install vosk`
- Download a model from: https://alphacephei.com/vosk/models
- Supports WAV format (mono, 16kHz recommended)

### Whisper Backend

**Advantages:**
- ✅ High accuracy
- ✅ Multiple audio formats supported
- ✅ Timestamps included

**Limitations:**
- ❌ Requires Python < 3.14 (numba dependency)
- ❌ Larger model sizes
- ❌ Slower transcription

## Implementation Details

The `AudioTranscriber` class is fully implemented with:
- ✅ `transcribe()` method for basic audio-to-text conversion
- ✅ `transcribe_with_timestamps()` method for segmented output with timestamps
- ✅ Comprehensive error handling for:
  - Missing files (`FileNotFoundError`)
  - Unsupported formats (`UnsupportedFormatError`)
  - Transcription failures (`TranscriptionError`)
- ✅ Support for multiple audio formats: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.webm`
- ✅ Graceful handling when whisper library is unavailable

## Quick Start with Vosk (Python 3.14)

### Step 1: Install Vosk

```bash
pip install vosk
```

### Step 2: Download a Model

```bash
# Use the helper script
python download_vosk_model.py

# Or manually download from: https://alphacephei.com/vosk/models
# Recommended: vosk-model-small-en-us-0.15 (40MB, good balance)
```

### Step 3: Use AudioTranscriber

```python
from src.loaders import AudioTranscriber

# Auto-select backend (will use Vosk if available)
transcriber = AudioTranscriber(backend="auto")

# Or explicitly use Vosk
transcriber = AudioTranscriber(
    backend="vosk",
    model_path="./models/vosk-model-small-en-us-0.15"
)

# Transcribe audio
text = transcriber.transcribe("audio.wav")
print(text)
```

### Audio Format Requirements for Vosk

Vosk works best with:
- Format: WAV
- Channels: Mono (1 channel)
- Sample rate: 16kHz (recommended)

To convert audio files:
```bash
# Using ffmpeg
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Testing

The implementation includes comprehensive tests:

### Unit Tests (Automated)
```bash
pytest tests/test_audio_transcriber.py -v
```

Tests will skip whisper-dependent tests when the library is unavailable, but will verify:
- Class structure and supported formats
- Error handling when whisper is not available

### Manual Tests
```bash
python tests/manual/test_audio_manual.py
```

Manual tests require:
1. Working whisper installation
2. Sample audio files

## Usage Examples

### Automatic Backend Selection (Recommended)

```python
from src.loaders import AudioTranscriber

# Automatically select best available backend
transcriber = AudioTranscriber(backend="auto")

# Basic transcription
text = transcriber.transcribe("lecture.wav")
print(text)

# Transcription with timestamps
segments = transcriber.transcribe_with_timestamps("lecture.wav")
for segment in segments:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

### Using Vosk Backend (Python 3.14 Compatible)

```python
from src.loaders import AudioTranscriber

# Download model first using: python download_vosk_model.py
transcriber = AudioTranscriber(
    backend="vosk",
    model_path="./models/vosk-model-small-en-us-0.15"
)

# Transcribe WAV file
text = transcriber.transcribe("lecture.wav")
print(text)
```

### Using Whisper Backend (Python < 3.14)

```python
from src.loaders import AudioTranscriber

# Use Whisper backend explicitly
transcriber = AudioTranscriber(
    backend="whisper",
    model_name="base"  # Options: tiny, base, small, medium, large
)

# Transcribe (supports multiple formats)
text = transcriber.transcribe("lecture.mp3")
print(text)
```

## Requirements Validation

The implementation satisfies all requirements from the design document:

- ✅ **Requirement 2.1**: Transcribes audio files to text
- ✅ **Requirement 2.2**: Processes supported audio formats
- ✅ **Requirement 2.3**: Returns full text transcript
- ✅ **Requirement 2.4**: Reports errors with specific details

## Next Steps

1. **For immediate use**: Switch to Python 3.13 or earlier
2. **For production**: Monitor numba/whisper compatibility updates
3. **Alternative**: Consider implementing a wrapper that supports multiple transcription backends
