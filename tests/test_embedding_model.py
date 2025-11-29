"""Unit tests for the EmbeddingModel class."""

import pytest
import numpy as np
from src.embeddings import EmbeddingModel, EmbeddingError, ModelLoadError


class TestEmbeddingModel:
    """Test suite for EmbeddingModel."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create an EmbeddingModel instance for testing."""
        return EmbeddingModel()
    
    def test_embedding_generation_with_sample_text(self, embedding_model):
        """Test that embedding generation works with sample text."""
        text = "This is a sample text for testing embeddings."
        embedding = embedding_model.embed(text)
        
        # Check that we got an embedding
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embedding_with_multiple_chunks(self, embedding_model):
        """Test batch embedding with multiple text chunks."""
        texts = [
            "First chunk of text.",
            "Second chunk of text.",
            "Third chunk of text."
        ]
        embeddings = embedding_model.embed_batch(texts)
        
        # Check that we got embeddings for all texts
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_embedding_dimensions_are_768(self, embedding_model):
        """Verify that embedding dimensions are 768 for all-mpnet-base-v2."""
        text = "Test text for dimension verification."
        embedding = embedding_model.embed(text)
        
        # all-mpnet-base-v2 produces 768-dimensional embeddings
        assert len(embedding) == 768
        
        # Also verify through the property
        assert embedding_model.embedding_dimension == 768
    
    def test_consistency_same_text_produces_same_embedding(self, embedding_model):
        """Test that the same text produces the same embedding."""
        text = "Consistency test text."
        
        # Generate embedding twice
        embedding1 = embedding_model.embed(text)
        embedding2 = embedding_model.embed(text)
        
        # Check that embeddings are identical
        assert len(embedding1) == len(embedding2)
        
        # Use numpy for numerical comparison with tolerance
        np.testing.assert_allclose(
            embedding1, 
            embedding2, 
            rtol=1e-7,
            err_msg="Same text should produce identical embeddings"
        )
    
    def test_empty_text_raises_error(self, embedding_model):
        """Test that empty text raises an EmbeddingError."""
        with pytest.raises(EmbeddingError, match="Cannot embed empty"):
            embedding_model.embed("")
    
    def test_whitespace_only_text_raises_error(self, embedding_model):
        """Test that whitespace-only text raises an EmbeddingError."""
        with pytest.raises(EmbeddingError, match="Cannot embed empty"):
            embedding_model.embed("   \n\t  ")
    
    def test_batch_with_empty_list(self, embedding_model):
        """Test that batch embedding with empty list returns empty list."""
        embeddings = embedding_model.embed_batch([])
        assert embeddings == []
    
    def test_batch_with_empty_text_raises_error(self, embedding_model):
        """Test that batch embedding with empty text raises an error."""
        texts = ["Valid text", "", "Another valid text"]
        with pytest.raises(EmbeddingError, match="Cannot embed empty"):
            embedding_model.embed_batch(texts)
    
    def test_different_texts_produce_different_embeddings(self, embedding_model):
        """Test that different texts produce different embeddings."""
        text1 = "Machine learning is fascinating."
        text2 = "The weather is nice today."
        
        embedding1 = embedding_model.embed(text1)
        embedding2 = embedding_model.embed(text2)
        
        # Embeddings should be different
        assert embedding1 != embedding2
        
        # Calculate cosine similarity to ensure they're not too similar
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        # Different texts should have lower similarity (not close to 1.0)
        assert similarity < 0.95
    
    def test_batch_embedding_dimensions_consistency(self, embedding_model):
        """Test that all embeddings in a batch have the same dimensions."""
        texts = [
            "Short text.",
            "This is a much longer text with more words and content.",
            "Medium length text here."
        ]
        embeddings = embedding_model.embed_batch(texts)
        
        # All embeddings should have the same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert all(dim == 768 for dim in dimensions)
    
    def test_invalid_model_name_raises_error(self):
        """Test that an invalid model name raises a ModelLoadError."""
        with pytest.raises(ModelLoadError, match="Failed to load model"):
            EmbeddingModel(model_name="invalid-model-name-12345")
