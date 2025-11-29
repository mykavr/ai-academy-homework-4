"""Embedding model for generating vector representations of text."""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class EmbeddingModel:
    """
    Generates vector embeddings from text using sentence-transformers.
    
    Uses the all-mpnet-base-v2 model which produces 768-dimensional embeddings.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            
        Raises:
            ModelLoadError: If the model fails to load
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model '{model_name}': {str(e)}"
            )
    
    def embed(self, text: str) -> List[float]:
        """
        Generate a vector embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embedding for text: {str(e)}"
            )
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A list of embedding vectors, one for each input text
            
        Raises:
            EmbeddingError: If embedding generation fails for any text
        """
        if not texts:
            return []
        
        # Check for empty texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmbeddingError(
                    f"Cannot embed empty or whitespace-only text at index {i}"
                )
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate batch embeddings: {str(e)}"
            )
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.
        
        Returns:
            The embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
