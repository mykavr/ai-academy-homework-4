"""Test script to verify Vosk setup and model."""

from src.loaders import AudioTranscriber, TranscriptionError
from src.config import default_config
from pathlib import Path


def test_vosk_model():
    """Test that Vosk model is properly configured."""
    print("=" * 70)
    print("VOSK SETUP VERIFICATION")
    print("=" * 70)
    
    # Check if model exists
    model_path = Path(default_config.vosk_model_path)
    print(f"\n1. Checking model path: {model_path}")
    
    if not model_path.exists():
        print(f"   ✗ Model not found at: {model_path}")
        print(f"\n   Please run: python download_vosk_model.py")
        return False
    
    print(f"   ✓ Model found at: {model_path}")
    
    # Try to initialize AudioTranscriber with Vosk backend
    print(f"\n2. Initializing AudioTranscriber with Vosk backend...")
    
    try:
        transcriber = AudioTranscriber(
            backend="vosk",
            model_path=str(model_path)
        )
        print(f"   ✓ AudioTranscriber initialized successfully")
        print(f"   Backend: {transcriber.backend_name}")
        print(f"   Supported formats: {transcriber.SUPPORTED_FORMATS}")
        
    except TranscriptionError as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False
    
    # Test with default config
    print(f"\n3. Testing with default config...")
    
    try:
        transcriber = AudioTranscriber(
            backend=default_config.transcription_backend,
            model_path=default_config.vosk_model_path
        )
        print(f"   ✓ Default config works")
        print(f"   Backend: {default_config.transcription_backend}")
        print(f"   Model path: {default_config.vosk_model_path}")
        
    except TranscriptionError as e:
        print(f"   ✗ Failed with default config: {e}")
        return False
    
    # Test auto backend selection
    print(f"\n4. Testing auto backend selection...")
    
    try:
        transcriber = AudioTranscriber(backend="auto")
        print(f"   ✓ Auto backend selection works")
        print(f"   Selected backend: {transcriber.backend_name}")
        
    except TranscriptionError as e:
        print(f"   ✗ Auto backend failed: {e}")
        # This is not critical, continue
    
    print("\n" + "=" * 70)
    print("✓ VOSK SETUP VERIFIED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Your Vosk backend is ready to use!")
    print()
    print("Example usage:")
    print("  from src.loaders import AudioTranscriber")
    print("  from src.config import default_config")
    print()
    print("  transcriber = AudioTranscriber(")
    print("      backend=default_config.transcription_backend,")
    print("      model_path=default_config.vosk_model_path")
    print("  )")
    print("  text = transcriber.transcribe('audio.wav')")
    print()
    
    return True


if __name__ == "__main__":
    success = test_vosk_model()
    exit(0 if success else 1)
