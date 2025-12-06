# Design Document

## Overview

This design implements lazy loading for heavy resources in the RAG chatbot system. The core strategy is to defer initialization of expensive components (Vosk models, embedding models, transcribers, processors) until they are first accessed. This is achieved through property-based lazy initialization patterns where components are stored as private attributes (initially None) and initialized through getter properties that check if the component exists before returning it.

The design maintains backward compatibility with existing code while significantly improving startup performance for lightweight operations.

## Architecture

### Current Architecture Issues

The current `RAGChatbot.__init__()` method eagerly initializes all components:
- PDF loader
- Audio transcriber (loads Vosk model)
- Video processor (loads Vosk model)
- Text chunker (loads tokenizer)
- Embedding model (loads sentence-transformers model)
- Vector store
- LLM interface

This means even running `python main.py stats` loads multi-gigabyte models into memory.

### Proposed Architecture

The refactored architecture uses lazy initialization:

1. **Initialization Phase**: Store configuration parameters only
2. **Access Phase**: Initialize components on first access through properties
3. **Caching Phase**: Reuse initialized components for subsequent operations

```
┌─────────────────────────────────────────┐
│         RAGChatbot.__init__()           │
│  - Store config parameters only         │
│  - Set all component refs to None       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      Component Access (Property)        │
│  - Check if component is None           │
│  - If None: initialize and cache        │
│  - Return cached component              │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       Subsequent Accesses               │
│  - Return cached component directly     │
└─────────────────────────────────────────┘
```

## Components and Interfaces

### RAGChatbot Refactoring

The `RAGChatbot` class will be refactored to use lazy properties:

**Private Attributes** (initialized to None):
- `_pdf_loader`
- `_audio_transcriber`
- `_video_processor`
- `_text_chunker`
- `_embedding_model`
- `_vector_store`
- `_llm`

**Configuration Attributes** (stored in `__init__`):
- `_model_path`
- `_persist_directory`
- `_collection_name`
- `_chunk_size`
- `_chunk_overlap`
- `_embedding_model_name`
- `_lm_studio_url`
- `_llm_timeout`
- `_llm_debug_logging`

**Lazy Properties** (with `@property` decorator):
- `pdf_loader` - initializes PDFLoader on first access
- `audio_transcriber` - initializes AudioTranscriber with Vosk model on first access
- `video_processor` - initializes VideoProcessor with Vosk model on first access
- `text_chunker` - initializes TextChunker on first access
- `embedding_model` - initializes EmbeddingModel on first access
- `vector_store` - initializes VectorStore on first access
- `llm` - initializes LLMInterface on first access

### Component Loading Strategy by Command

| Command | Components Loaded |
|---------|------------------|
| stats | vector_store only |
| clear | vector_store only |
| ask | embedding_model, vector_store, llm |
| ingest (PDF) | pdf_loader, text_chunker, embedding_model, vector_store |
| ingest (audio) | audio_transcriber, text_chunker, embedding_model, vector_store |
| ingest (video) | video_processor, text_chunker, embedding_model, vector_store |
| interactive | embedding_model, vector_store, llm |

## Data Models

No changes to data models are required. The refactoring is purely architectural and maintains the same interfaces.

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

After analyzing the acceptance criteria, several properties are redundant or can be consolidated. The reflection below identifies these redundancies:

**Property Reflection:**
- Properties 1.1, 1.2, 1.3, 1.4 all test that stats doesn't load various components - these can be consolidated into one property about stats not loading heavy resources
- Property 4.1 is a duplicate of 1.2
- Properties 2.4, 2.5, 2.6 all test that ask doesn't load transcription components - can be consolidated
- Properties 3.1, 3.2, 3.3 all test that PDF ingestion doesn't load transcription components - can be consolidated
- Properties 5.2 and 5.3 both test caching behavior - can be consolidated

**Consolidated Properties:**

Property 1: Stats command loads only vector store
*For any* RAG chatbot instance, when the stats command is executed, only the vector store component should be initialized, and all other heavy components (Vosk model, audio transcriber, video processor, embedding model) should remain uninitialized
**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 4.1**

Property 2: Ask command loads only query components
*For any* RAG chatbot instance, when the ask command is executed, only the embedding model, vector store, and LLM interface should be initialized, and transcription components (Vosk model, audio transcriber, video processor) should remain uninitialized
**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**

Property 3: PDF ingestion excludes transcription components
*For any* RAG chatbot instance, when a PDF file is ingested, the transcription components (Vosk model, audio transcriber, video processor) should remain uninitialized
**Validates: Requirements 3.1, 3.2, 3.3**

Property 4: Audio ingestion loads Vosk model
*For any* RAG chatbot instance, when an audio file is ingested, the Vosk model and audio transcriber should be initialized
**Validates: Requirements 3.4**

Property 5: Video ingestion loads Vosk model
*For any* RAG chatbot instance, when a video file is ingested, the Vosk model and video processor should be initialized
**Validates: Requirements 3.5**

Property 6: Clear command loads only vector store
*For any* RAG chatbot instance, when the clear command is executed, only the vector store should be initialized, and the embedding model should remain uninitialized
**Validates: Requirements 4.2**

Property 7: Document ingestion loads embedding model
*For any* RAG chatbot instance, when any document (PDF, audio, or video) is ingested, the embedding model should be initialized
**Validates: Requirements 4.3**

Property 8: Question answering loads embedding model
*For any* RAG chatbot instance, when a question is asked, the embedding model should be initialized
**Validates: Requirements 4.4**

Property 9: Component caching preserves identity
*For any* RAG chatbot instance and any component, when that component is accessed multiple times, all accesses should return the same object instance (verified by identity check)
**Validates: Requirements 4.5, 5.2, 5.3**

Property 10: Unused components remain uninitialized
*For any* RAG chatbot instance, when a component is never accessed through its property, that component should remain None
**Validates: Requirements 5.4**

## Error Handling

### Lazy Initialization Errors

When a component fails to initialize during lazy loading:
1. Catch the initialization exception
2. Wrap it in a descriptive error that indicates which component failed
3. Include the original error message
4. Propagate the error to the caller

Example:
```python
@property
def audio_transcriber(self):
    if self._audio_transcriber is None:
        try:
            self._audio_transcriber = AudioTranscriber(model_path=self._model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize audio transcriber: {str(e)}"
            ) from e
    return self._audio_transcriber
```

### Backward Compatibility

The refactoring maintains backward compatibility:
- All public methods remain unchanged
- Component access through properties is transparent to callers
- Error messages are enhanced but error types remain consistent

## Testing Strategy

### Unit Testing

Unit tests will verify:
1. Each lazy property initializes its component correctly on first access
2. Each lazy property returns the same instance on subsequent accesses (caching behavior)
3. Initialization errors are properly wrapped and propagated with clear messages
4. Configuration parameters are correctly passed to components during lazy initialization
5. Components remain uninitialized (None) when not accessed
6. Different command operations load only their required components

**Testing Approach:**
- Use mocking to avoid actually loading heavy models during testing
- Verify component initialization state by checking if private attributes are None or initialized
- Test each lazy property in isolation
- Test command methods (stats, ask, ingest_pdf, ingest_audio, ingest_video, clear) to verify they load only required components
- Use object identity checks (using `is` operator) to verify caching behavior
