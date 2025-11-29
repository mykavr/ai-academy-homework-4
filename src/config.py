"""Configuration settings for the RAG chatbot system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration for RAG chatbot system."""
    
    # Text chunking parameters
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 75  # tokens (15% of chunk_size)
    
    # Embedding model
    embedding_model: str = "all-mpnet-base-v2"
    embedding_dimension: int = 768
    
    # Vector database
    vector_db_path: str = "./chroma_db"
    
    # LM Studio configuration
    lm_studio_url: str = "http://localhost:1234/v1"
    
    # Retrieval parameters
    top_k: int = 5
    
    # Whisper model for audio transcription
    whisper_model: str = "base"  # Options: tiny, base, small, medium, large
    
    # File size limits (in MB)
    max_pdf_size: int = 100
    max_audio_size: int = 500
    max_video_size: int = 1000


# Default configuration instance
default_config = RAGConfig()
