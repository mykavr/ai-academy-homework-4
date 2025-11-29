"""Vector store implementation using Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import uuid
import logging

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Exception raised for storage-related errors."""
    pass


class VectorStore:
    """Vector store for storing and querying document embeddings using Qdrant."""
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "documents"):
        """
        Initialize Qdrant vector store with optional persistent storage.
        
        Args:
            persist_directory: Directory path for persistent storage (None for in-memory)
            collection_name: Name of the collection to use
        """
        try:
            # Initialize client (in-memory if no path provided)
            if persist_directory:
                self.client = QdrantClient(path=persist_directory)
            else:
                self.client = QdrantClient(":memory:")
            
            self.collection_name = collection_name
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logger.info(f"Created new collection '{collection_name}'")
            else:
                logger.info(f"Using existing collection '{collection_name}'")
            
            storage_type = "persistent" if persist_directory else "in-memory"
            logger.info(f"Initialized VectorStore with {storage_type} storage")
            
        except Exception as e:
            raise StorageError(f"Failed to initialize Qdrant: {str(e)}")
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Store embeddings with their associated text and metadata.
        
        Args:
            texts: List of text chunks to store
            embeddings: List of embedding vectors corresponding to texts
            metadata: Optional list of metadata dictionaries for each text
            
        Raises:
            StorageError: If storage operation fails
            ValueError: If input lists have mismatched lengths
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of embeddings ({len(embeddings)})"
            )
        
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError(
                f"Number of metadata entries ({len(metadata)}) must match number of texts ({len(texts)})"
            )
        
        if not texts:
            logger.warning("add_documents called with empty texts list")
            return
        
        try:
            # Prepare metadata (use empty dict if not provided)
            if metadata is None:
                metadata = [{} for _ in range(len(texts))]
            
            # Create points for Qdrant
            points = []
            for i in range(len(texts)):
                point_id = str(uuid.uuid4())
                
                # Combine text with metadata in payload
                payload = {
                    "text": texts[i],
                    **metadata[i]  # Merge metadata into payload
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=embeddings[i],
                    payload=payload
                )
                points.append(point)
            
            # Upsert points to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully added {len(texts)} documents to vector store")
            
        except Exception as e:
            raise StorageError(f"Failed to add documents to vector store: {str(e)}")
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve top-k most similar documents based on query embedding.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing:
                - text: The document text
                - metadata: Associated metadata
                - distance: Similarity distance (lower is more similar)
                
        Raises:
            StorageError: If query operation fails
        """
        try:
            # Query the collection (note: Qdrant uses 'query' not 'search' for the method name)
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k
            ).points
            
            # Format results
            formatted_results = []
            for result in search_results:
                # Extract text from payload
                text = result.payload.get("text", "")
                
                # Extract metadata (all payload fields except 'text')
                metadata = {k: v for k, v in result.payload.items() if k != "text"}
                
                formatted_result = {
                    'text': text,
                    'metadata': metadata,
                    'distance': 1.0 - result.score  # Convert similarity to distance (lower is better)
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Query returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            raise StorageError(f"Failed to query vector store: {str(e)}")
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        
        Raises:
            StorageError: If clear operation fails
        """
        try:
            # Delete all points from the collection
            # Using scroll to get all point IDs, then delete them
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Get up to 10000 points at once
                with_payload=False,
                with_vectors=False
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
            
            logger.info(f"Cleared all documents from collection '{self.collection_name}'")
            
        except Exception as e:
            raise StorageError(f"Failed to clear vector store: {str(e)}")
    
    def count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents stored
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return collection_info.points_count
        except Exception as e:
            raise StorageError(f"Failed to count documents: {str(e)}")
    
    def close(self) -> None:
        """
        Close the Qdrant client and release resources.
        Important for persistent storage to avoid file locking issues.
        """
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {str(e)}")
