# Audio Transcription Guide

## Overview

The AudioTranscriber now supports multiple backends with automatic selection, making it compatible with Python 3.14 and providing flexibility in choosing transcription engines.

## Architecture

```
AudioTranscriber (Wrapper)
    ├── Backend: Auto (automatically selects best available)
    ├── Backend: Vosk (Python 3.14 compatible, offline, fast)
    └── Backend: Whisper (High accuracy, requires Python < 3.14)
```

## Quick Start

### 1. Install Dependencies

```bash
# For Python 3.14 (recommended)
pip install vosk

# For Python < 3.14 (optional)
pip install openai-whisper
```

### 2. Download Vosk Model (if using Vosk)

```bash
python download_vosk_model.py
```

This will download a small English model (~40MB) to `./models/`

### 3. Use AudioTranscriber

```python
from src.loaders import AudioTranscriber

# Automatic backend selection (recommended)
transcriber = AudioTranscriber(backend="auto")

# Transcribe audio
text = transcriber.transcribe("audio.wav")
print(text)
```

## Backend Comparison

| Feature | Vosk | Whisper |
|---------|------|---------|
| Python 3.14 Support | ✅ Yes | ❌ No (requires < 3.14) |
| Offline | ✅ Yes | ✅ Yes |
| Speed | ✅ Fast | ⚠️ Slower |
| Accuracy | ⚠️ Good | ✅ Excellent |
| Audio Formats | WAV only | MP3, WAV, M4A, FLAC, OGG, OPUS, WebM |
| Model Size | 40MB - 1.8GB | 140MB - 2.9GB |
| Setup Complexity | Medium (need to download model) | Easy (auto-downloads) |

## Usage Examples

### Example 1: Auto Backend (Recommended)

```python
from src.loaders import AudioTranscriber

# Let the system choose the best available backend
transcriber = AudioTranscriber(backend="auto")

# Check which backend was selected
print(f"Using: {transcriber.backend_name}")

# Transcribe
text = transcriber.transcribe("lecture.wav")
print(text)
```

### Example 2: Vosk Backend (Python 3.14)

```python
from src.loaders import AudioTranscriber

# Explicitly use Vosk
transcriber = AudioTranscriber(
    backend="vosk",
    model_path="./models/vosk-model-small-en-us-0.15"
)

# Basic transcription
text = transcriber.transcribe("lecture.wav")

# With timestamps
segments = transcriber.transcribe_with_timestamps("lecture.wav")
for seg in segments:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
```

### Example 3: Whisper Backend (Python < 3.14)

```python
from src.loaders import AudioTranscriber

# Explicitly use Whisper
transcriber = AudioTranscriber(
    backend="whisper",
    model_name="base"  # Options: tiny, base, small, medium, large
)

# Supports multiple formats
text = transcriber.transcribe("lecture.mp3")
print(text)
```

## Audio Format Requirements

### For Vosk:
- **Format**: WAV
- **Channels**: Mono (1 channel)
- **Sample Rate**: 16kHz (recommended)

Convert audio files:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### For Whisper:
- **Formats**: MP3, WAV, M4A, FLAC, OGG, OPUS, WebM
- **No specific requirements** (handles conversion internally)

## Configuration

Update `src/config.py` to set default backend:

```python
@dataclass
class RAGConfig:
    # Audio transcription backend
    transcription_backend: str = "auto"  # Options: auto, whisper, vosk
    
    # Whisper model
    whisper_model: str = "base"
    
    # Vosk model path
    vosk_model_path: str = "./models/vosk-model-small-en-us-0.15"
```

## Troubleshooting

### Issue: "No transcription backend available"

**Solution**: Install at least one backend:
```bash
pip install vosk  # Recommended for Python 3.14
# OR
pip install openai-whisper  # For Python < 3.14
```

### Issue: "Vosk model not found"

**Solution**: Download a model:
```bash
python download_vosk_model.py
```

Or manually download from: https://alphacephei.com/vosk/models

### Issue: "Audio must be mono (1 channel)"

**Solution**: Convert to mono:
```bash
ffmpeg -i input.wav -ac 1 output.wav
```

### Issue: Whisper not working on Python 3.14

**Solution**: Use Vosk backend instead:
```python
transcriber = AudioTranscriber(backend="vosk", model_path="./models/vosk-model-small-en-us-0.15")
```

## Testing

Run the test suite:
```bash
pytest tests/test_audio_transcriber.py -v
```

Run examples:
```bash
python examples/audio_transcription_example.py
```

## Files Created

- `src/loaders/transcription_backends.py` - Backend implementations
- `src/loaders/audio_transcriber.py` - Main wrapper class
- `tests/test_audio_transcriber.py` - Unit tests
- `download_vosk_model.py` - Helper script to download Vosk models
- `examples/audio_transcription_example.py` - Usage examples
- `AUDIO_TRANSCRIPTION_GUIDE.md` - This guide
- `WHISPER_COMPATIBILITY.md` - Updated compatibility notes

## Next Steps

1. Download a Vosk model: `python download_vosk_model.py`
2. Test with your audio files
3. Integrate into the RAG pipeline (Task 9)
4. Consider adding more backends (e.g., faster-whisper, whisper.cpp)
