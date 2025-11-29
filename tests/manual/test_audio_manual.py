"""Manual test for AudioTranscriber - requires actual audio files.

This test is meant to be run manually when you have sample audio files to test with.

Requirements:
1. Vosk model downloaded (run: python download_vosk_model.py)
2. Sample audio files in WAV format (mono, 16kHz recommended)

Usage:
    python tests/manual/test_audio_manual.py [path/to/audio.wav]
    
Example:
    python tests/manual/test_audio_manual.py "path/to/audio.wav"
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.loaders import AudioTranscriber, TranscriptionError, UnsupportedFormatError
from src.config import default_config


def test_audio_transcription(audio_file=None):
    """Manual test for audio transcription."""
    print("=" * 70)
    print("MANUAL AUDIO TRANSCRIPTION TEST")
    print("=" * 70)
    
    try:
        # Initialize transcriber with configured backend
        print(f"\n1. Initializing AudioTranscriber...")
        print(f"   Backend: {default_config.transcription_backend}")
        print(f"   Model path: {default_config.vosk_model_path}")
        
        transcriber = AudioTranscriber(
            backend=default_config.transcription_backend,
            model_path=default_config.vosk_model_path
        )
        print("   ✓ AudioTranscriber initialized successfully")
        print(f"   Backend: {transcriber.backend_name}")
        print(f"   Supported formats: {transcriber.SUPPORTED_FORMATS}")
        
        # Get audio file from argument or use default
        if audio_file is None:
            audio_file = "path/to/your/audio/file.wav"
        
        if not Path(audio_file).exists():
            print(f"\n⚠ Audio file not found: {audio_file}")
            print("   Please provide an audio file path as an argument:")
            print('   python tests/manual/test_audio_manual.py "path/to/audio.wav"')
            return
        
        # Test basic transcription
        print(f"\n2. Transcribing audio file: {audio_file}")
        text = transcriber.transcribe(audio_file)
        print("   ✓ Transcription completed")
        print(f"\n   Transcribed text:\n   {text[:200]}...")  # Show first 200 chars
        
        # Test transcription with timestamps
        print(f"\n3. Transcribing with timestamps...")
        segments = transcriber.transcribe_with_timestamps(audio_file)
        print(f"   ✓ Transcription with timestamps completed")
        print(f"   Number of segments: {len(segments)}")
        
        if segments:
            print("\n   First segment:")
            print(f"   - Start: {segments[0]['start']:.2f}s")
            print(f"   - End: {segments[0]['end']:.2f}s")
            print(f"   - Text: {segments[0]['text']}")
        
        print("\n" + "=" * 70)
        print("SUCCESS: All manual tests completed")
        print("=" * 70)
        
    except TranscriptionError as e:
        print(f"\n✗ TranscriptionError: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


def test_error_handling():
    """Test error handling without requiring audio files."""
    print("\n" + "=" * 70)
    print("ERROR HANDLING TESTS")
    print("=" * 70)
    
    try:
        transcriber = AudioTranscriber(
            backend=default_config.transcription_backend,
            model_path=default_config.vosk_model_path
        )
        
        # Test file not found
        print("\n1. Testing FileNotFoundError...")
        try:
            transcriber.transcribe("nonexistent_file.mp3")
            print("   ✗ Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            print(f"   ✓ FileNotFoundError raised correctly: {e}")
        
        # Test unsupported format
        print("\n2. Testing UnsupportedFormatError...")
        try:
            # Create a temporary text file
            temp_file = Path("temp_test.txt")
            temp_file.write_text("test")
            
            transcriber.transcribe(str(temp_file))
            print("   ✗ Should have raised UnsupportedFormatError")
        except UnsupportedFormatError as e:
            print(f"   ✓ UnsupportedFormatError raised correctly")
            print(f"   Message: {e}")
        finally:
            if temp_file.exists():
                temp_file.unlink()
        
        print("\n" + "=" * 70)
        print("SUCCESS: Error handling tests completed")
        print("=" * 70)
        
    except TranscriptionError as e:
        if "model" in str(e).lower():
            print(f"\n⚠ Model not available: {e}")
            print("\nTo download a Vosk model:")
            print("  python download_vosk_model.py")
        else:
            print(f"\n✗ TranscriptionError: {e}")


if __name__ == "__main__":
    # Get audio file from command line argument if provided
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if audio_file:
        print(f"Using audio file: {audio_file}\n")
    
    test_error_handling()
    print("\n")
    test_audio_transcription(audio_file)
