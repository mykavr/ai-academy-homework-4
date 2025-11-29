"""Integration test script for the RAG pipeline.

This script tests the complete flow: PDF loading → chunking → embedding → vector storage.

Usage:
    python tests/manual/test_integration.py <path_to_pdf_file>

Example:
    python tests/manual/test_integration.py sample.pdf
    python tests/manual/test_integration.py "C:/Documents/my document.pdf"
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all components
from src.loaders import PDFLoader, PDFProcessingError
from src.processing import TextChunker
from src.embeddings import EmbeddingModel, EmbeddingError, ModelLoadError
from src.storage import VectorStore, StorageError


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_step(step_num: int, description: str):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {description}")
    print("-"*70)


def print_success(message: str):
    """Print a success message."""
    print(f"[OK] {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"[ERROR] {message}")


def print_info(label: str, value):
    """Print an info line."""
    print(f"  • {label}: {value}")


def step1_load_pdf(pdf_path: str) -> Dict:
    """Step 1: Load PDF and extract text."""
    print_step(1, "Loading PDF")
    
    try:
        loader = PDFLoader()
        result = loader.load_with_metadata(pdf_path)
        
        print_success(f"PDF loaded successfully")
        print_info("Source", result['source'])
        print_info("Total pages", result['num_pages'])
        print_info("Total characters", len(result['text']))
        print_info("Total lines", len(result['text'].splitlines()))
        
        # Show preview
        preview_length = 200
        preview = result['text'][:preview_length]
        print(f"\n  Preview (first {preview_length} chars):")
        print(f"  {preview}...")
        
        return result
        
    except (FileNotFoundError, PDFProcessingError) as e:
        print_error(f"PDF loading failed: {e}")
        raise
    except Exception as e:
        print_error(f"Unexpected error during PDF loading: {e}")
        raise


def step2_chunk_text(pdf_result: Dict) -> List[Dict]:
    """Step 2: Chunk the extracted text."""
    print_step(2, "Chunking text")
    
    try:
        chunker = TextChunker(chunk_size=512, chunk_overlap=75)
        
        chunks = chunker.chunk_with_metadata(
            text=pdf_result['text'],
            source=pdf_result['source']
        )
        
        print_success(f"Text chunked successfully")
        print_info("Total chunks", len(chunks))
        print_info("Chunk size (tokens)", 512)
        print_info("Chunk overlap (tokens)", 75)
        
        # Show all chunks with full text
        if chunks:
            print(f"\n  All chunks (full text):")
            print("  " + "-"*68)
            for i, chunk in enumerate(chunks):
                # Calculate actual token count for this chunk
                token_count = len(chunker.tokenizer.encode(chunk['text']))
                
                print(f"\n  [Chunk {i}]")
                print(f"  Source: {chunk['source']}")
                print(f"  Chunk index: {chunk['chunk_index']} of {chunk['total_chunks']}")
                print(f"  Text length: {len(chunk['text'])} characters")
                print(f"  Token count: {token_count} tokens")
                print(f"\n  Text:")
                # Indent each line of the chunk text
                for line in chunk['text'].split('\n'):
                    print(f"    {line}")
                print("  " + "-"*68)
        
        return chunks
        
    except Exception as e:
        print_error(f"Text chunking failed: {e}")
        raise


def step3_generate_embeddings(chunks: List[Dict]) -> List[List[float]]:
    """Step 3: Generate embeddings for all chunks."""
    print_step(3, "Generating embeddings")
    
    try:
        embedding_model = EmbeddingModel(model_name="all-mpnet-base-v2")
        
        # Extract just the text from chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        print_info("Model", "all-mpnet-base-v2")
        print_info("Embedding dimension", embedding_model.embedding_dimension)
        print(f"  • Generating embeddings for {len(chunk_texts)} chunks...")
        
        embeddings = embedding_model.embed_batch(chunk_texts)
        
        print_success(f"Embeddings generated successfully")
        print_info("Total embeddings", len(embeddings))
        print_info("Embedding shape", f"{len(embeddings)} x {len(embeddings[0])}")
        
        # Show sample embedding stats
        if embeddings:
            sample_embedding = embeddings[0]
            avg_value = sum(sample_embedding) / len(sample_embedding)
            min_value = min(sample_embedding)
            max_value = max(sample_embedding)
            print(f"\n  Sample embedding statistics:")
            print_info("  Average value", f"{avg_value:.4f}")
            print_info("  Min value", f"{min_value:.4f}")
            print_info("  Max value", f"{max_value:.4f}")
        
        return embeddings
        
    except (ModelLoadError, EmbeddingError) as e:
        print_error(f"Embedding generation failed: {e}")
        raise
    except Exception as e:
        print_error(f"Unexpected error during embedding generation: {e}")
        raise


def step4_store_vectors(chunks: List[Dict], embeddings: List[List[float]]) -> VectorStore:
    """Step 4: Store embeddings in vector database."""
    print_step(4, "Storing vectors in database")
    
    try:
        # Use in-memory storage for testing
        vector_store = VectorStore(persist_directory=None, collection_name="test_integration")
        
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadata = [
            {
                'source': chunk['source'],
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks']
            }
            for chunk in chunks
        ]
        
        print_info("Storage type", "in-memory")
        print_info("Collection name", "test_integration")
        print(f"  • Storing {len(texts)} documents...")
        
        vector_store.add_documents(texts=texts, embeddings=embeddings, metadata=metadata)
        
        print_success(f"Vectors stored successfully")
        print_info("Documents in store", vector_store.count())
        
        return vector_store
        
    except (StorageError, ValueError) as e:
        print_error(f"Vector storage failed: {e}")
        raise
    except Exception as e:
        print_error(f"Unexpected error during vector storage: {e}")
        raise


def step5_verify_storage(vector_store: VectorStore, chunks: List[Dict], embeddings: List[List[float]]):
    """Step 5: Verify storage by reading back and performing a test query."""
    print_step(5, "Verifying storage")
    
    try:
        # Check document count
        doc_count = vector_store.count()
        print_info("Documents in database", doc_count)
        
        if doc_count != len(chunks):
            print_error(f"Document count mismatch! Expected {len(chunks)}, got {doc_count}")
            return
        
        print_success("Document count matches")
        
        # Perform a test query using the first chunk's embedding
        if embeddings:
            print(f"\n  • Performing test query with first chunk...")
            query_results = vector_store.query(query_embedding=embeddings[0], top_k=3)
            
            print_success(f"Query returned {len(query_results)} results")
            
            # Display query results with full text
            print(f"\n  Top {len(query_results)} similar documents (full text):")
            print("  " + "-"*68)
            for i, result in enumerate(query_results):
                distance = result['distance']
                chunk_idx = result['metadata'].get('chunk_index', 'N/A')
                source = result['metadata'].get('source', 'N/A')
                total_chunks = result['metadata'].get('total_chunks', 'N/A')
                
                print(f"\n  [Result {i+1}]")
                print(f"  Source: {source}")
                print(f"  Chunk index: {chunk_idx} of {total_chunks}")
                print(f"  Distance: {distance:.4f}")
                print(f"  Text length: {len(result['text'])} characters")
                print(f"\n  Full text:")
                # Indent each line of the result text
                for line in result['text'].split('\n'):
                    print(f"    {line}")
                print("  " + "-"*68)
            
            # The first result should be the same chunk (distance ~0)
            if query_results and query_results[0]['distance'] < 0.01:
                print_success("Self-query verification passed (distance < 0.01)")
            else:
                print_error("Self-query verification failed (distance too high)")
        
    except StorageError as e:
        print_error(f"Storage verification failed: {e}")
        raise
    except Exception as e:
        print_error(f"Unexpected error during verification: {e}")
        raise


def step6_cleanup(vector_store: VectorStore):
    """Step 6: Clean up the vector database."""
    print_step(6, "Cleaning up")
    
    try:
        doc_count_before = vector_store.count()
        print_info("Documents before cleanup", doc_count_before)
        
        print("  • Clearing vector database...")
        vector_store.clear()
        
        doc_count_after = vector_store.count()
        print_info("Documents after cleanup", doc_count_after)
        
        if doc_count_after == 0:
            print_success("Database cleared successfully")
        else:
            print_error(f"Database not fully cleared ({doc_count_after} documents remaining)")
        
        # Close the vector store
        vector_store.close()
        print_success("Vector store closed")
        
    except StorageError as e:
        print_error(f"Cleanup failed: {e}")
        raise
    except Exception as e:
        print_error(f"Unexpected error during cleanup: {e}")
        raise


def print_summary(pdf_result: Dict, chunks: List[Dict], embeddings: List[List[float]]):
    """Print a summary of the integration test."""
    print_header("INTEGRATION TEST SUMMARY")
    
    print("\nPDF Processing:")
    print_info("  Source", pdf_result['source'])
    print_info("  Pages", pdf_result['num_pages'])
    print_info("  Characters", len(pdf_result['text']))
    
    print("\nText Chunking:")
    print_info("  Total chunks", len(chunks))
    print_info("  Chunk size", "512 tokens")
    print_info("  Overlap", "75 tokens")
    
    print("\nEmbeddings:")
    print_info("  Model", "all-mpnet-base-v2")
    print_info("  Dimension", len(embeddings[0]) if embeddings else 0)
    print_info("  Total embeddings", len(embeddings))
    
    print("\nVector Storage:")
    print_info("  Storage type", "in-memory")
    print_info("  Documents stored", len(chunks))
    print_info("  Status", "Cleaned up")
    
    print("\n" + "="*70)
    print("SUCCESS: ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*70)


def main():
    """Main entry point for the integration test."""
    if len(sys.argv) < 2:
        print("Usage: python tests/manual/test_integration.py <path_to_pdf_file>")
        print("\nExample:")
        print("  python tests/manual/test_integration.py sample.pdf")
        print('  python tests/manual/test_integration.py "C:/Documents/my document.pdf"')
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    print_header("RAG PIPELINE INTEGRATION TEST")
    print(f"Testing with file: {pdf_path}\n")
    
    try:
        # Step 1: Load PDF
        pdf_result = step1_load_pdf(pdf_path)
        
        # Step 2: Chunk text
        chunks = step2_chunk_text(pdf_result)
        
        # Step 3: Generate embeddings
        embeddings = step3_generate_embeddings(chunks)
        
        # Step 4: Store vectors
        vector_store = step4_store_vectors(chunks, embeddings)
        
        # Step 5: Verify storage
        step5_verify_storage(vector_store, chunks, embeddings)
        
        # Step 6: Cleanup
        step6_cleanup(vector_store)
        
        # Print summary
        print_summary(pdf_result, chunks, embeddings)
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR: INTEGRATION TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
