"""Text chunking module for splitting text into semantic chunks."""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


class TextChunker:
    """Handles text chunking using token-based splitting with overlap."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 75, 
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize TextChunker with specified parameters.
        
        Args:
            chunk_size: Maximum number of tokens per chunk (default: 512)
            chunk_overlap: Number of tokens to overlap between chunks (default: 75)
            model_name: Hugging Face model name for tokenizer (default: all-mpnet-base-v2)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Load tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create text splitter using the tokenizer
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token size.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def chunk_with_metadata(self, text: str, source: str) -> List[Dict[str, any]]:
        """
        Split text into chunks and include metadata about the source.
        
        Args:
            text: Input text to be chunked
            source: Source identifier (e.g., filename)
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            return []
        
        chunks = self.chunk_text(text)
        
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "text": chunk,
                "source": source,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
        
        return chunks_with_metadata
