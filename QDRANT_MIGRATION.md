# Vector Database Migration: ChromaDB → Qdrant

## Summary

Due to Python 3.14 compatibility issues with ChromaDB and its dependencies (particularly onnxruntime and pydantic v1), the project has been migrated to use **Qdrant** as the vector database solution.

## Compatibility Issues with ChromaDB

ChromaDB 0.5.0 and later versions have the following issues with Python 3.14:
- **Pydantic V1**: Core functionality isn't compatible with Python 3.14
- **onnxruntime**: No pre-built wheels available for Python 3.14
- **pypika**: Build failures due to deprecated AST attributes in Python 3.14
- **chroma-hnswlib**: No Python 3.14 support

## Qdrant Advantages

✅ **Full Python 3.14 compatibility** - All dependencies install and work correctly
✅ **Simple API** - Similar interface to ChromaDB with minimal code changes
✅ **In-memory and persistent storage** - Flexible deployment options
✅ **High performance** - Optimized for similarity search
✅ **Local execution** - No external dependencies or cloud services required
✅ **Active development** - Well-maintained with regular updates

## Installation

```bash
pip install qdrant-client==1.16.1
```

Successfully tested on Python 3.14.0 with Windows.

## API Comparison

### ChromaDB
```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("documents")
```

### Qdrant
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")  # or path for persistent storage
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
```

## Changes Made

### 1. requirements.txt
- **Before**: `chromadb==0.5.0`
- **After**: `qdrant-client==1.16.1`

### 2. design.md
- Updated Vector Database section to reference Qdrant
- Changed schema to use Qdrant's structure (vector + payload)
- Updated configuration example with `vector_db_path: "./qdrant_storage"`

### 3. tasks.md
- Updated task 5.1 to reference Qdrant initialization
- Added note about creating collection with 768-dimensional vectors

## Implementation Notes

When implementing the VectorStore class:
- Use `QdrantClient(":memory:")` for testing
- Use `QdrantClient(path="./qdrant_storage")` for persistent storage
- Create collection with `VectorParams(size=768, distance=Distance.COSINE)` to match all-mpnet-base-v2 embeddings
- Store text and metadata in the `payload` field
- Use `client.upsert()` to add documents
- Use `client.search()` for similarity queries

## Verification

Qdrant has been successfully tested with:
- ✅ Installation on Python 3.14.0
- ✅ Client creation (in-memory mode)
- ✅ Collection creation with 768-dimensional vectors
- ✅ COSINE distance metric configuration

## Next Steps

Continue with task 5.2 (Write unit tests for vector store) using Qdrant instead of ChromaDB.
