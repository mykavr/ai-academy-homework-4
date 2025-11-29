"""Example demonstrating audio transcription with different backends."""

from src.loaders import AudioTranscriber, TranscriptionError
from src.config import default_config
from pathlib import Path


def example_auto_backend():
    """Example using automatic backend selection."""
    print("=" * 70)
    print("EXAMPLE 1: Automatic Backend Selection")
    print("=" * 70)
    
    try:
        # Create transcriber with auto backend selection
        transcriber = AudioTranscriber(backend="auto")
        print(f"✓ Using backend: {transcriber.backend_name}")
        print(f"  Supported formats: {transcriber.SUPPORTED_FORMATS}")
        
        # Example transcription (you need to provide an actual audio file)
        audio_file = "path/to/your/audio.wav"
        
        if Path(audio_file).exists():
            print(f"\nTranscribing: {audio_file}")
            text = transcriber.transcribe(audio_file)
            print(f"Result: {text}")
        else:
            print(f"\n⚠ Audio file not found: {audio_file}")
            print("  Please provide a valid audio file path to test transcription.")
        
    except TranscriptionError as e:
        print(f"✗ Error: {e}")


def example_vosk_backend():
    """Example using Vosk backend explicitly."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Vosk Backend (Python 3.14 Compatible)")
    print("=" * 70)
    
    try:
        # Create transcriber with Vosk backend using configured model
        transcriber = AudioTranscriber(
            backend="vosk",
            model_path=default_config.vosk_model_path
        )
        print("✓ Vosk backend initialized")
        print(f"  Supported formats: {transcriber.SUPPORTED_FORMATS}")
        
        # Example transcription
        audio_file = "path/to/your/audio.wav"
        
        if Path(audio_file).exists():
            print(f"\nTranscribing: {audio_file}")
            
            # Basic transcription
            text = transcriber.transcribe(audio_file)
            print(f"\nBasic transcription:\n{text}")
            
            # Transcription with timestamps
            segments = transcriber.transcribe_with_timestamps(audio_file)
            print(f"\nTranscription with timestamps:")
            for i, segment in enumerate(segments[:3], 1):  # Show first 3 segments
                print(f"  Segment {i}: [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                print(f"    Text: {segment['text']}")
        else:
            print(f"\n⚠ Audio file not found: {audio_file}")
            print("  Please provide a valid WAV file to test transcription.")
            print("\n  Note: Vosk requires WAV format (mono, 16kHz recommended)")
            print("  Convert with: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav")
        
    except TranscriptionError as e:
        print(f"✗ Error: {e}")
        if "model" in str(e).lower():
            print("\n  To download a Vosk model, run:")
            print("    python download_vosk_model.py")


def example_whisper_backend():
    """Example using Whisper backend explicitly."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Whisper Backend (Requires Python < 3.14)")
    print("=" * 70)
    
    try:
        # Create transcriber with Whisper backend
        transcriber = AudioTranscriber(
            backend="whisper",
            model_name="base"
        )
        print("✓ Whisper backend initialized")
        print(f"  Supported formats: {transcriber.SUPPORTED_FORMATS}")
        
        # Example transcription
        audio_file = "path/to/your/audio.mp3"
        
        if Path(audio_file).exists():
            print(f"\nTranscribing: {audio_file}")
            
            # Basic transcription
            text = transcriber.transcribe(audio_file)
            print(f"\nBasic transcription:\n{text}")
            
            # Transcription with timestamps
            segments = transcriber.transcribe_with_timestamps(audio_file)
            print(f"\nTranscription with timestamps:")
            for i, segment in enumerate(segments[:3], 1):  # Show first 3 segments
                print(f"  Segment {i}: [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                print(f"    Text: {segment['text']}")
        else:
            print(f"\n⚠ Audio file not found: {audio_file}")
            print("  Please provide a valid audio file to test transcription.")
        
    except TranscriptionError as e:
        print(f"✗ Error: {e}")
        if "not available" in str(e):
            print("\n  Whisper requires Python < 3.14 due to numba dependency.")
            print("  Consider using Vosk backend instead for Python 3.14.")


def example_error_handling():
    """Example demonstrating error handling."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Error Handling")
    print("=" * 70)
    
    try:
        transcriber = AudioTranscriber(backend="auto")
        
        # Test file not found
        print("\n1. Testing FileNotFoundError:")
        try:
            transcriber.transcribe("nonexistent_file.wav")
        except FileNotFoundError as e:
            print(f"   ✓ Caught: {e}")
        
        # Test unsupported format
        print("\n2. Testing UnsupportedFormatError:")
        try:
            # Create a temporary text file
            temp_file = Path("temp_test.txt")
            temp_file.write_text("test")
            transcriber.transcribe(str(temp_file))
        except Exception as e:
            print(f"   ✓ Caught: {type(e).__name__}: {e}")
        finally:
            if temp_file.exists():
                temp_file.unlink()
        
        print("\n✓ Error handling works correctly")
        
    except TranscriptionError as e:
        print(f"✗ Could not initialize transcriber: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("AUDIO TRANSCRIPTION EXAMPLES")
    print("=" * 70)
    print()
    print("This script demonstrates the AudioTranscriber with different backends.")
    print()
    
    # Run examples
    example_auto_backend()
    example_vosk_backend()
    example_whisper_backend()
    example_error_handling()
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Download a Vosk model: python download_vosk_model.py")
    print("  2. Update audio file paths in this script")
    print("  3. Run again to see actual transcription results")
    print()


if __name__ == "__main__":
    main()
