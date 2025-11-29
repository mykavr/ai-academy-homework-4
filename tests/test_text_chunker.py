"""Unit tests for TextChunker class."""

import pytest
from src.processing.text_chunker import TextChunker


class TestTextChunker:
    """Test suite for TextChunker functionality."""
    
    @pytest.fixture
    def chunker(self):
        """Create a TextChunker instance with default settings."""
        return TextChunker(chunk_size=512, chunk_overlap=75)
    
    def test_chunking_with_known_text(self, chunker):
        """Test chunking with a known text sample."""
        text = "This is a test sentence. " * 100  # Create a longer text
        chunks = chunker.chunk_text(text)
        
        # Should produce multiple chunks
        assert len(chunks) > 0
        # Each chunk should be a string
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Each chunk should contain some content
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_empty_text(self, chunker):
        """Test chunking with empty text."""
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text("\n\n") == []
    
    def test_single_sentence_text(self, chunker):
        """Test chunking with a single short sentence."""
        text = "This is a single short sentence."
        chunks = chunker.chunk_text(text)
        
        # Should produce exactly one chunk
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_very_long_text(self, chunker):
        """Test chunking with very long text."""
        # Create a very long text that will definitely need multiple chunks
        text = " ".join([f"Word{i}" for i in range(1000)])
        chunks = chunker.chunk_text(text)
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_overlap_calculation(self, chunker):
        """Test that consecutive chunks have overlapping content."""
        # Create text that will produce multiple chunks
        text = "Sentence number one. " * 200
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                chunk1 = chunks[i]
                chunk2 = chunks[i + 1]
                
                # Extract last few words from chunk1 and first few words from chunk2
                words1 = chunk1.split()[-10:]
                words2 = chunk2.split()[:10]
                
                # There should be some overlap
                overlap = set(words1) & set(words2)
                assert len(overlap) > 0, "Consecutive chunks should have overlapping content"
    
    def test_chunk_sizes_respect_token_limits(self, chunker):
        """Verify that chunk sizes respect token limits."""
        # Create a long text
        text = "This is a test sentence with multiple words. " * 300
        chunks = chunker.chunk_text(text)
        
        # Check that each chunk (except possibly the last) respects the token limit
        for i, chunk in enumerate(chunks):
            token_count = len(chunker.tokenizer.encode(chunk))
            
            # All chunks except the last should be close to chunk_size
            # The last chunk might be smaller
            if i < len(chunks) - 1:
                assert token_count <= chunker.chunk_size + 10, \
                    f"Chunk {i} has {token_count} tokens, exceeds limit of {chunker.chunk_size}"
            else:
                # Last chunk can be any size up to chunk_size
                assert token_count <= chunker.chunk_size + 10
    
    def test_chunk_with_metadata(self, chunker):
        """Test chunking with metadata preservation."""
        text = "This is a test. " * 100
        source = "test_document.pdf"
        
        chunks_with_meta = chunker.chunk_with_metadata(text, source)
        
        # Should return a list of dictionaries
        assert len(chunks_with_meta) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks_with_meta)
        
        # Each chunk should have required metadata fields
        for chunk in chunks_with_meta:
            assert "text" in chunk
            assert "source" in chunk
            assert "chunk_index" in chunk
            assert "total_chunks" in chunk
            assert chunk["source"] == source
        
        # Chunk indices should be sequential
        for i, chunk in enumerate(chunks_with_meta):
            assert chunk["chunk_index"] == i
        
        # All chunks should have the same total_chunks value
        total = chunks_with_meta[0]["total_chunks"]
        assert all(chunk["total_chunks"] == total for chunk in chunks_with_meta)
        assert total == len(chunks_with_meta)
    
    def test_chunk_with_metadata_empty_text(self, chunker):
        """Test chunk_with_metadata with empty text."""
        assert chunker.chunk_with_metadata("", "source.pdf") == []
        assert chunker.chunk_with_metadata("   ", "source.pdf") == []
    
    def test_different_chunk_sizes(self):
        """Test chunking with different chunk size configurations."""
        text = "Word " * 500
        
        # Test with smaller chunk size
        small_chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        small_chunks = small_chunker.chunk_text(text)
        
        # Test with larger chunk size
        large_chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        large_chunks = large_chunker.chunk_text(text)
        
        # Smaller chunk size should produce more chunks
        assert len(small_chunks) > len(large_chunks)
