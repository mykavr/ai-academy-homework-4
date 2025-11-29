# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that processes multi-format knowledge bases (PDFs, audio, and video files) and enables users to ask questions about the content.

## Features

- **Multi-format ingestion**: Process PDFs, audio files, and video files
- **Local execution**: All processing happens locally with no cloud dependencies
- **Semantic search**: Vector-based retrieval for accurate context matching
- **LLM-powered answers**: Generate responses using local LLM via LM Studio

## Prerequisites

- Python 3.8 or higher
- LM Studio installed and running locally
- FFmpeg (for audio/video processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up LM Studio:
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Download a local model (e.g., Llama 2 7B, Mistral 7B)
   - Start the local server (default: http://localhost:1234)

## Project Structure

```
rag-chatbot/
├── src/
│   ├── loaders/          # PDF, audio, video loaders
│   ├── processing/       # Text chunking
│   ├── embeddings/       # Vector embedding generation
│   ├── storage/          # Vector database (ChromaDB)
│   ├── rag/              # RAG pipeline and LLM interface
│   └── config.py         # Configuration settings
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
└── main.py              # CLI entry point
```

## Configuration

Default configuration is defined in `src/config.py`:

- **Chunk size**: 512 tokens
- **Chunk overlap**: 75 tokens (15%)
- **Embedding model**: all-mpnet-base-v2 (768 dimensions)
- **Top-k retrieval**: 5 chunks
- **Whisper model**: base
- **LM Studio URL**: http://localhost:1234/v1

## Usage

(Usage instructions will be added as features are implemented)

## Development

Run tests:
```bash
pytest tests/
```

## License

MIT License
