# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that processes multi-format knowledge bases (PDFs, audio, and video files) and enables users to ask questions about the content. The system runs entirely locally with no cloud dependencies.

## Features

- **Multi-format ingestion**: Process PDFs, audio files (WAV, MP3, FLAC, etc.), and video files (MP4, AVI, MOV, etc.)
- **Local execution**: All processing happens locally with no cloud dependencies or API keys required
- **Semantic search**: Vector-based retrieval using 768-dimensional embeddings for accurate context matching
- **LLM-powered answers**: Generate responses using local LLM via LM Studio
- **Interactive mode**: Chat with your knowledge base in real-time
- **Flexible CLI**: Command-line interface for ingestion, querying, and knowledge base management

## Prerequisites

- **Python 3.8 or higher** (tested with Python 3.14)
- **LM Studio** installed and running locally
- **FFmpeg** (for audio/video processing)
- **Vosk model** for speech recognition (automatically downloaded on first use)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd rag-chatbot
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

FFmpeg is required for audio and video processing.

**Windows:**
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add to PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg      # CentOS/RHEL
```

### 4. Download Vosk model

The Vosk speech recognition model will be automatically downloaded on first use. Alternatively, you can download it manually:

```bash
python download_vosk_model.py
```

This downloads the `vosk-model-small-en-us-0.15` model (~40MB) to the `models/` directory.

### 5. Set up LM Studio

LM Studio provides the local LLM for answer generation.

1. **Download and install** [LM Studio](https://lmstudio.ai/)

2. **Download a model** (recommended models):
   - **Llama 2 7B** - Good balance of quality and speed
   - **Mistral 7B** - High quality responses
   - **Phi-2** - Faster, smaller model
   - Any other model compatible with LM Studio

3. **Start the local server**:
   - Open LM Studio
   - Load your chosen model
   - Go to the "Local Server" tab
   - Click "Start Server"
   - Default URL: `http://localhost:1234/v1`
   - Verify the server is running (you should see a green indicator)

4. **Test the connection** (optional):
   ```bash
   python tests/manual/test_lm_studio_connection.py
   ```

## Project Structure

```
rag-chatbot/
├── src/
│   ├── loaders/          # PDF, audio, video loaders
│   │   ├── pdf_loader.py
│   │   ├── audio_transcriber.py
│   │   ├── video_processor.py
│   │   └── transcription_backends.py
│   ├── processing/       # Text chunking
│   │   └── text_chunker.py
│   ├── embeddings/       # Vector embedding generation
│   │   └── embedding_model.py
│   ├── storage/          # Vector database (Qdrant)
│   │   └── vector_store.py
│   ├── rag/              # RAG pipeline and LLM interface
│   │   ├── chatbot.py
│   │   └── llm_interface.py
│   └── config.py         # Configuration settings
├── tests/                # Test suite
│   ├── test_*.py         # Unit tests
│   └── manual/           # Manual integration tests
├── models/               # Vosk speech recognition models
├── examples/             # Example scripts
├── requirements.txt      # Python dependencies
├── main.py              # CLI entry point
└── README.md            # This file
```

## Configuration

The system can be configured by modifying `src/config.py` or using command-line arguments.

### Default Configuration

```python
# Text chunking parameters
chunk_size: 512           # tokens per chunk
chunk_overlap: 75         # token overlap between chunks (15%)

# Embedding model
embedding_model: "all-mpnet-base-v2"  # 768-dimensional embeddings
embedding_dimension: 768

# Vector database
vector_db_path: "./qdrant_storage"

# LM Studio configuration
lm_studio_url: "http://localhost:1234/v1"

# Retrieval parameters
top_k: 5                  # number of context chunks to retrieve

# Vosk model path
vosk_model_path: "models/vosk-model-small-en-us-0.15"

# File size limits (in MB)
max_pdf_size: 100
max_audio_size: 500
max_video_size: 1000
```

### Configuration Options

You can override configuration using command-line arguments:

```bash
# Use custom database path
python main.py --db-path ./my_custom_db ingest document.pdf

# Use custom LM Studio URL
python main.py --lm-studio-url http://localhost:5000/v1 ask "What is this about?"

# Use custom Vosk model
python main.py --model-path ./models/vosk-model-en-us-0.22 ingest audio.wav
```

## Usage

The RAG chatbot provides a command-line interface with several commands.

### Basic Command Structure

```bash
python main.py [global-options] <command> [command-options]
```

### Commands

#### 1. Ingest Documents

Add documents to your knowledge base.

**Ingest a PDF:**
```bash
python main.py ingest document.pdf
```

**Ingest an audio file:**
```bash
python main.py ingest lecture.wav
```

**Ingest a video file:**
```bash
python main.py ingest presentation.mp4
```

**Specify file type explicitly:**
```bash
python main.py ingest myfile --type audio
```

**Supported formats:**
- **PDF**: `.pdf`
- **Audio**: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`

#### 2. Ask Questions

Query your knowledge base with a question.

**Basic question:**
```bash
python main.py ask "What is the main topic of the lecture?"
```

**Retrieve more context chunks:**
```bash
python main.py ask "Explain the key concepts" --top-k 10
```

**Hide context chunks in output:**
```bash
python main.py ask "Summarize the content" --no-context
```

#### 3. Interactive Mode

Start an interactive chat session with your knowledge base.

```bash
python main.py interactive
```

**Interactive commands:**
- Type your question and press Enter
- Type `stats` to see knowledge base statistics
- Type `quit`, `exit`, or `q` to leave

**Example session:**
```
You: What are the main topics covered?
Bot: The main topics covered include...

You: Tell me more about topic X
Bot: Topic X is discussed in detail...

You: stats
📊 Knowledge Base Statistics:
   Total chunks: 245
   Collection: rag_documents

You: quit
👋 Goodbye!
```

#### 4. View Statistics

Display information about your knowledge base.

```bash
python main.py stats
```

Output:
```
📊 KNOWLEDGE BASE STATISTICS
Total chunks: 245
Collection: rag_documents
```

#### 5. Clear Knowledge Base

Remove all documents from the knowledge base.

```bash
python main.py clear
```

You will be prompted for confirmation before clearing.

### Complete Usage Examples

**Example 1: Process a lecture PDF and ask questions**
```bash
# Ingest the PDF
python main.py ingest lecture_notes.pdf

# Ask a question
python main.py ask "What are the key takeaways from this lecture?"

# Start interactive mode
python main.py interactive
```

**Example 2: Process multiple audio recordings**
```bash
# Ingest multiple files
python main.py ingest lecture1.wav
python main.py ingest lecture2.mp3
python main.py ingest lecture3.flac

# Check statistics
python main.py stats

# Ask questions
python main.py ask "Compare the topics discussed in the lectures"
```

**Example 3: Process a video presentation**
```bash
# Ingest video (audio will be extracted and transcribed)
python main.py ingest presentation.mp4

# Ask about the content
python main.py ask "What was demonstrated in the presentation?" --top-k 8
```

**Example 4: Custom configuration**
```bash
# Use custom database and LM Studio URL
python main.py --db-path ./project_kb --lm-studio-url http://localhost:5000/v1 ingest docs.pdf
python main.py --db-path ./project_kb --lm-studio-url http://localhost:5000/v1 ask "Summarize the project"
```

## How It Works

The RAG chatbot follows a multi-stage pipeline:

### Ingestion Pipeline

1. **Load**: Extract content from PDF, audio, or video files
   - PDFs: Extract text using pdfplumber
   - Audio: Transcribe using Vosk speech recognition
   - Video: Extract audio track, then transcribe

2. **Chunk**: Split text into semantic chunks
   - Uses RecursiveCharacterTextSplitter with 512 token chunks
   - 75 token overlap to preserve context
   - Respects sentence boundaries

3. **Embed**: Convert chunks to vector embeddings
   - Uses sentence-transformers with all-mpnet-base-v2 model
   - Generates 768-dimensional embeddings

4. **Store**: Save embeddings to vector database
   - Uses Qdrant for efficient similarity search
   - Stores metadata (source, chunk index, type)

### Query Pipeline

1. **Embed Question**: Convert user question to vector embedding

2. **Retrieve**: Find most similar chunks from vector database
   - Uses cosine similarity
   - Returns top-k most relevant chunks (default: 5)

3. **Generate**: Send question + context to LLM
   - Formats prompt with retrieved context
   - LM Studio generates answer based on context

4. **Return**: Display answer with sources and context

## Development

### Running Tests

**Run all unit tests:**
```bash
pytest tests/
```

**Run specific test file:**
```bash
pytest tests/test_text_chunker.py
```

**Run with verbose output:**
```bash
pytest tests/ -v
```

### Manual Testing

Manual integration tests are available in `tests/manual/`:

```bash
# Test LM Studio connection
python tests/manual/test_lm_studio_connection.py

# Test LLM interface
python tests/manual/test_llm_manual.py

# Test audio transcription
python tests/manual/test_audio_manual.py

# Test full RAG pipeline
python tests/manual/test_rag_chatbot_manual.py
```

## Troubleshooting

### LM Studio Connection Issues

**Problem**: "Failed to connect to LM Studio"

**Solutions**:
1. Verify LM Studio is running and server is started
2. Check the server URL (default: `http://localhost:1234/v1`)
3. Ensure a model is loaded in LM Studio
4. Test connection: `python tests/manual/test_lm_studio_connection.py`

### Audio Transcription Issues

**Problem**: "Vosk model not found"

**Solutions**:
1. Run `python download_vosk_model.py` to download the model
2. Verify model exists in `models/vosk-model-small-en-us-0.15/`
3. Check model path in configuration

**Problem**: "FFmpeg not found"

**Solutions**:
1. Install FFmpeg (see Installation section)
2. Verify FFmpeg is in PATH: `ffmpeg -version`

### Memory Issues

**Problem**: Out of memory during processing

**Solutions**:
1. Process smaller files or split large files
2. Reduce chunk size in configuration
3. Use a smaller embedding model
4. Close other applications to free memory

### Empty or Poor Quality Answers

**Problem**: Chatbot returns "I don't have enough information"

**Solutions**:
1. Verify documents were ingested: `python main.py stats`
2. Increase top-k parameter: `--top-k 10`
3. Check if question is related to ingested content
4. Try rephrasing the question

**Problem**: Low quality or irrelevant answers

**Solutions**:
1. Use a better LLM model in LM Studio
2. Increase context chunks: `--top-k 8`
3. Ensure ingested documents are relevant
4. Check LM Studio model is properly loaded

## Performance Considerations

- **First run**: Initial model downloads may take time (Vosk ~40MB, embeddings ~400MB)
- **PDF ingestion**: ~1-5 seconds per page
- **Audio transcription**: ~1-2x real-time (10 min audio = 10-20 min processing)
- **Video processing**: Similar to audio (depends on audio length)
- **Question answering**: ~2-10 seconds (depends on LLM model and hardware)

## System Requirements

**Minimum:**
- 8GB RAM
- 5GB free disk space
- Dual-core CPU

**Recommended:**
- 16GB RAM
- 10GB free disk space
- Quad-core CPU
- GPU (optional, speeds up embedding generation)

## Privacy and Security

- **Fully local**: All processing happens on your machine
- **No cloud services**: No data sent to external APIs
- **No API keys**: No authentication or API keys required
- **Offline capable**: Works without internet after initial setup
