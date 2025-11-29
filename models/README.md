# Vosk Models Directory

This directory contains Vosk speech recognition models. Models are **not included in the repository** due to their large size.

## Download Models

### Option 1: Use the Download Script (Recommended)

Run the provided script from the project root:

```bash
python download_vosk_model.py
```

This will download the small English model (~40MB) to this directory.

### Option 2: Manual Download

1. Visit: https://alphacephei.com/vosk/models
2. Download a model (recommended: `vosk-model-small-en-us-0.15`)
3. Extract the zip file to this directory

### Available Models

**Small Models (Recommended for most use cases):**
- `vosk-model-small-en-us-0.15` - 40MB, good accuracy, fast
- `vosk-model-small-en-us-0.22` - 40MB, updated version

**Large Models (Better accuracy, slower):**
- `vosk-model-en-us-0.22` - 1.8GB, best accuracy

**Other Languages:**
Visit https://alphacephei.com/vosk/models for models in other languages.

## Current Configuration

The project is configured to use:
- Model: `vosk-model-small-en-us-0.15`
- Path: `models/vosk-model-small-en-us-0.15`

You can change this in `src/config.py`:

```python
vosk_model_path: str = "models/vosk-model-small-en-us-0.15"
```

## Verify Installation

After downloading, verify the model works:

```bash
python test_vosk_setup.py
```

Expected output:
```
✓ VOSK SETUP VERIFIED SUCCESSFULLY
```

## Directory Structure

After downloading, your directory should look like:

```
models/
├── README.md (this file)
└── vosk-model-small-en-us-0.15/
    ├── am/
    ├── conf/
    ├── graph/
    ├── ivector/
    └── ...
```

## Troubleshooting

### Model not found error

If you get "Vosk model not found" error:
1. Ensure the model is extracted to the correct path
2. Check that the directory name matches the configuration
3. Run `python test_vosk_setup.py` to diagnose

### Download failed

If the download script fails:
1. Check your internet connection
2. Try manual download from the website
3. Ensure you have write permissions to this directory

## Notes

- Models are excluded from git via `.gitignore`
- Each developer needs to download models separately
- Models are cached and don't need to be re-downloaded
- Different models can coexist in this directory
