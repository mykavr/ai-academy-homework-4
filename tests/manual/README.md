# Manual Integration Tests

This directory contains manual integration tests for the RAG pipeline.

## test_integration.py

A comprehensive integration test that validates the complete RAG pipeline flow for any supported file type.

### Supported File Types

- **PDF**: `.pdf`
- **Audio**: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.webm`, `.wma`, `.aiff`, `.aif`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpg`, `.mpeg`

### Pipeline Flow

The test validates the following steps:

1. **File Loading/Processing**
   - PDF: Extract text from PDF pages
   - Audio: Transcribe audio to text using Vosk
   - Video: Extract audio track and transcribe to text

2. **Text Chunking**
   - Split text into semantic chunks (512 tokens with 75 token overlap)
   - Preserve metadata (source, chunk index)

3. **Embedding Generation**
   - Generate vector embeddings using all-mpnet-base-v2 model
   - 768-dimensional embeddings

4. **Vector Storage**
   - Store embeddings in Qdrant vector database
   - Include metadata for retrieval

5. **Storage Verification**
   - Verify document count
   - Perform test query to validate similarity search

6. **Cleanup**
   - Clear vector database
   - Close connections

### Usage

```bash
# Test with PDF
python tests/manual/test_integration.py sample.pdf

# Test with audio
python tests/manual/test_integration.py lecture.mp3

# Test with video
python tests/manual/test_integration.py recording.mp4

# Test with file path containing spaces
python tests/manual/test_integration.py "C:/Documents/my document.pdf"
```

### Requirements

- All dependencies from `requirements.txt` must be installed
- Vosk model must be downloaded (for audio/video processing)
- ffmpeg must be installed (for audio format conversion and video processing)

### Output

The test provides detailed output for each step:
- File loading/processing statistics
- Chunk information with full text
- Embedding statistics
- Storage verification results
- Final summary

### Example Output

```
======================================================================
RAG PIPELINE INTEGRATION TEST
======================================================================
Testing with file: lecture.mp3
Detected file type: AUDIO

[Step 1] Transcribing Audio
----------------------------------------------------------------------
[OK] Audio transcribed successfully
  • Source: lecture.mp3
  • File type: Audio
  • Total characters: 1234
  • Total lines: 5

  Preview (first 200 chars):
  This is a lecture about machine learning...

[Step 2] Chunking text
----------------------------------------------------------------------
[OK] Text chunked successfully
  • Total chunks: 3
  • Chunk size (tokens): 512
  • Chunk overlap (tokens): 75

...

======================================================================
SUCCESS: ALL STEPS COMPLETED SUCCESSFULLY
======================================================================
```
