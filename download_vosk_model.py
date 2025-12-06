"""Helper script to download a Vosk model for speech recognition."""

import os
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_vosk_model(model_name="vosk-model-small-en-us-0.15", models_dir="./models"):
    """
    Download a Vosk model.
    
    Args:
        model_name: Name of the model to download
                   Options:
                   - vosk-model-small-en-us-0.15 (40MB, fast, good accuracy)
                   - vosk-model-en-us-0.22 (1.8GB, best accuracy)
        models_dir: Directory to store models
    """
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    model_path = models_path / model_name
    
    # Check if model already exists
    if model_path.exists():
        print(f"✓ Model already exists at: {model_path}")
        return str(model_path)
    
    # Model URLs
    model_urls = {
        "vosk-model-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "vosk-model-en-us-0.22": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "vosk-model-small-en-us-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.22.zip",
        "vosk-model-en-us-0.42-gigaspeech": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
    }
    
    if model_name not in model_urls:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(model_urls.keys())}")
        return None
    
    url = model_urls[model_name]
    zip_path = models_path / f"{model_name}.zip"
    
    print(f"Downloading {model_name}...")
    print(f"URL: {url}")
    
    try:
        # Download the model
        download_url(url, zip_path)
        
        # Extract the model
        print(f"\nExtracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_path)
        
        # Remove the zip file
        zip_path.unlink()
        
        print(f"✓ Model downloaded and extracted to: {model_path}")
        return str(model_path)
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("VOSK MODEL DOWNLOADER")
    print("=" * 70)
    print()
    print("This script will download a Vosk speech recognition model.")
    print()
    print("Available models:")
    print("  1. vosk-model-small-en-us-0.15 (40MB, recommended)")
    print("  2. vosk-model-en-us-0.22 (1.8GB, best accuracy)")
    print("  3. vosk-model-en-us-0.42-gigaspeech (2.3GB)")
    print()
    
    choice = input("Select model (1, 2, or 3, default=1): ").strip() or "1"
    
    if choice == "1":
        model_name = "vosk-model-small-en-us-0.15"
    elif choice == "2":
        model_name = "vosk-model-en-us-0.22"
    elif choice == "3":
        model_name = "vosk-model-en-us-0.42-gigaspeech"
    else:
        print("Invalid choice. Using default model.")
        model_name = "vosk-model-small-en-us-0.15"
    
    print()
    model_path = download_vosk_model(model_name)
    
    if model_path:
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print()
        print("To use this model with AudioTranscriber:")
        print()
        print("  from src.loaders import AudioTranscriber")
        print(f"  transcriber = AudioTranscriber(backend='vosk', model_path='{model_path}')")
        print("  text = transcriber.transcribe('audio.wav')")
        print()
