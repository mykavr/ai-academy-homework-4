"""Unit tests for the VectorStore class."""

import pytest
import tempfile
import shutil
import os
from src.storage import VectorStore, StorageError


class TestVectorStore:
    """Test suite for VectorStore."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def vector_store(self, temp_db_path):
        """Create a VectorStore instance for testing."""
        store = VectorStore(persist_directory=temp_db_path, collection_name="test_collection")
        yield store
        # Close the client to release file locks
        store.close()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing."""
        # Create simple 768-dimensional embeddings (matching all-mpnet-base-v2)
        return [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ]
    
    def test_adding_and_retrieving_documents(self, vector_store, sample_embeddings):
        """Test that documents can be added and retrieved."""
        texts = [
            "First document about machine learning.",
            "Second document about artificial intelligence.",
            "Third document about data science."
        ]
        
        # Add documents
        vector_store.add_documents(texts=texts, embeddings=sample_embeddings)
        
        # Verify documents were added
        assert vector_store.count() == 3
        
        # Query with the first embedding (should return similar documents)
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=3)
        
        # Check that we got results
        assert len(results) == 3
        assert all('text' in result for result in results)
        assert all('metadata' in result for result in results)
        assert all('distance' in result for result in results)
    
    def test_metadata_preservation_with_specific_examples(self, vector_store, sample_embeddings):
        """Test that metadata is preserved when storing and retrieving documents."""
        texts = [
            "Document one content.",
            "Document two content.",
            "Document three content."
        ]
        
        metadata = [
            {"source": "file1.pdf", "chunk_index": 0, "source_type": "pdf"},
            {"source": "file2.pdf", "chunk_index": 1, "source_type": "pdf"},
            {"source": "audio1.mp3", "chunk_index": 0, "source_type": "audio"}
        ]
        
        # Add documents with metadata
        vector_store.add_documents(texts=texts, embeddings=sample_embeddings, metadata=metadata)
        
        # Query and verify metadata is preserved
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=3)
        
        # Check that all results have metadata
        assert len(results) == 3
        for result in results:
            assert 'metadata' in result
            assert 'source' in result['metadata']
            assert 'chunk_index' in result['metadata']
            assert 'source_type' in result['metadata']
            
            # Verify metadata values are from our original set
            assert result['metadata']['source_type'] in ['pdf', 'audio']
    
    def test_incremental_additions_without_data_loss(self, vector_store, sample_embeddings):
        """Test that adding new documents doesn't remove existing ones."""
        # First batch
        texts1 = ["First batch document one.", "First batch document two."]
        embeddings1 = sample_embeddings[:2]
        metadata1 = [
            {"source": "batch1_file1.pdf", "batch": 1},
            {"source": "batch1_file2.pdf", "batch": 1}
        ]
        
        vector_store.add_documents(texts=texts1, embeddings=embeddings1, metadata=metadata1)
        count_after_first = vector_store.count()
        assert count_after_first == 2
        
        # Second batch
        texts2 = ["Second batch document one."]
        embeddings2 = [sample_embeddings[2]]
        metadata2 = [{"source": "batch2_file1.pdf", "batch": 2}]
        
        vector_store.add_documents(texts=texts2, embeddings=embeddings2, metadata=metadata2)
        count_after_second = vector_store.count()
        
        # Verify no data loss
        assert count_after_second == 3
        
        # Query to verify all documents are still accessible
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=3)
        assert len(results) == 3
        
        # Verify we have documents from both batches
        batches = [result['metadata'].get('batch') for result in results]
        assert 1 in batches
        assert 2 in batches
    
    def test_database_clearing_functionality(self, vector_store, sample_embeddings):
        """Test that the clear method removes all documents."""
        texts = ["Document one.", "Document two.", "Document three."]
        
        # Add documents
        vector_store.add_documents(texts=texts, embeddings=sample_embeddings)
        assert vector_store.count() == 3
        
        # Clear the database
        vector_store.clear()
        
        # Verify database is empty
        assert vector_store.count() == 0
        
        # Verify queries return no results
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=5)
        assert len(results) == 0
    
    def test_query_result_limiting_with_various_top_k_values(self, vector_store, sample_embeddings):
        """Test that query respects the top_k parameter."""
        # Add 5 documents
        texts = [f"Document number {i}." for i in range(5)]
        embeddings = [sample_embeddings[0]] * 5  # Use same embedding for simplicity
        
        vector_store.add_documents(texts=texts, embeddings=embeddings)
        
        # Test various top_k values
        for k in [1, 2, 3, 5]:
            results = vector_store.query(query_embedding=sample_embeddings[0], top_k=k)
            assert len(results) == k, f"Expected {k} results, got {len(results)}"
        
        # Test top_k larger than available documents
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=10)
        assert len(results) == 5  # Should return all 5 available documents
    
    def test_empty_texts_list(self, vector_store):
        """Test that adding empty texts list doesn't cause errors."""
        # Should not raise an error, just log a warning
        vector_store.add_documents(texts=[], embeddings=[])
        assert vector_store.count() == 0
    
    def test_mismatched_texts_and_embeddings_raises_error(self, vector_store, sample_embeddings):
        """Test that mismatched texts and embeddings raises ValueError."""
        texts = ["Document one.", "Document two."]
        embeddings = sample_embeddings  # 3 embeddings
        
        with pytest.raises(ValueError, match="must match number of embeddings"):
            vector_store.add_documents(texts=texts, embeddings=embeddings)
    
    def test_mismatched_metadata_raises_error(self, vector_store, sample_embeddings):
        """Test that mismatched metadata length raises ValueError."""
        texts = ["Document one.", "Document two.", "Document three."]
        metadata = [{"source": "file1.pdf"}, {"source": "file2.pdf"}]  # Only 2 metadata entries
        
        with pytest.raises(ValueError, match="must match number of texts"):
            vector_store.add_documents(texts=texts, embeddings=sample_embeddings, metadata=metadata)
    
    def test_add_documents_without_metadata(self, vector_store, sample_embeddings):
        """Test that documents can be added without metadata."""
        texts = ["Document one.", "Document two.", "Document three."]
        
        # Add without metadata
        vector_store.add_documents(texts=texts, embeddings=sample_embeddings)
        
        # Verify documents were added
        assert vector_store.count() == 3
        
        # Query and verify empty metadata
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=3)
        assert len(results) == 3
        for result in results:
            assert 'metadata' in result
            assert result['metadata'] == {}
    
    def test_query_with_empty_database(self, vector_store, sample_embeddings):
        """Test that querying an empty database returns empty results."""
        results = vector_store.query(query_embedding=sample_embeddings[0], top_k=5)
        assert len(results) == 0
    
    def test_persistence_across_instances(self, temp_db_path, sample_embeddings):
        """Test that data persists across VectorStore instances."""
        texts = ["Persistent document one.", "Persistent document two."]
        embeddings = sample_embeddings[:2]
        metadata = [{"source": "persist1.pdf"}, {"source": "persist2.pdf"}]
        
        # Create first instance and add documents
        store1 = VectorStore(persist_directory=temp_db_path, collection_name="persist_test")
        store1.add_documents(texts=texts, embeddings=embeddings, metadata=metadata)
        count1 = store1.count()
        assert count1 == 2
        store1.close()  # Close to release file locks
        
        # Create second instance with same path
        store2 = VectorStore(persist_directory=temp_db_path, collection_name="persist_test")
        count2 = store2.count()
        
        # Verify data persisted
        assert count2 == 2
        
        # Verify we can query the persisted data
        results = store2.query(query_embedding=embeddings[0], top_k=2)
        assert len(results) == 2
        store2.close()  # Close to release file locks
    
    def test_different_collections_are_isolated(self, temp_db_path, sample_embeddings):
        """Test that different collections don't interfere with each other."""
        texts1 = ["Collection 1 document."]
        texts2 = ["Collection 2 document."]
        
        # Create first store and add documents
        store1 = VectorStore(persist_directory=temp_db_path, collection_name="collection1")
        store1.add_documents(texts=texts1, embeddings=[sample_embeddings[0]])
        assert store1.count() == 1
        store1.close()  # Close to release file locks
        
        # Create second store with different collection
        store2 = VectorStore(persist_directory=temp_db_path, collection_name="collection2")
        store2.add_documents(texts=texts2, embeddings=[sample_embeddings[1]])
        assert store2.count() == 1
        store2.close()  # Close to release file locks
        
        # Reopen first store and clear it
        store1 = VectorStore(persist_directory=temp_db_path, collection_name="collection1")
        store1.clear()
        assert store1.count() == 0
        store1.close()
        
        # Verify second collection is unaffected
        store2 = VectorStore(persist_directory=temp_db_path, collection_name="collection2")
        assert store2.count() == 1
        store2.close()
