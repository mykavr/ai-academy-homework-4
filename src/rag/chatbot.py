"""
RAG Chatbot orchestration module.

This module orchestrates the complete RAG pipeline:
1. Ingestion: Load documents from various sources (PDF, audio, video)
2. Processing: Chunk text into semantic segments
3. Embedding: Convert chunks to vector embeddings
4. Storage: Store embeddings in vector database
5. Query: Answer questions using retrieval and generation
"""

from typing import List, Dict, Optional
from pathlib import Path
import logging

from ..loaders.pdf_loader import PDFLoader, PDFProcessingError
from ..loaders.audio_transcriber import AudioTranscriber, TranscriptionError, UnsupportedFormatError
from ..loaders.video_processor import VideoProcessor, VideoProcessingError
from ..processing.text_chunker import TextChunker
from ..embeddings.embedding_model import EmbeddingModel, EmbeddingError, ModelLoadError
from ..storage.vector_store import VectorStore, StorageError
from .llm_interface import LLMInterface, LLMError

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Orchestrates the complete RAG pipeline for document ingestion and question answering.
    
    This class wires together all components:
    - Document loaders (PDF, audio, video)
    - Text chunking
    - Embedding generation
    - Vector storage
    - LLM-based answer generation
    """
    
    def __init__(
        self,
        model_path: str = None,
        persist_directory: Optional[str] = None,
        collection_name: str = "documents",
        chunk_size: int = 512,
        chunk_overlap: int = 75,
        embedding_model: str = "all-mpnet-base-v2",
        lm_studio_url: str = "http://localhost:1234/v1",
        llm_timeout: int = 60,
        llm_debug_logging: bool = False,
        top_k: int = 5
    ):
        """
        Initialize the RAG chatbot with configuration parameters only.
        Components are initialized lazily on first access.
        
        Args:
            model_path: Path to Vosk model for audio transcription (uses config default if not specified)
            persist_directory: Directory for persistent vector storage (None for in-memory)
            collection_name: Name of the vector store collection
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            embedding_model: Name of the sentence-transformers model
            lm_studio_url: URL for LM Studio server
            llm_timeout: Maximum wait time for LLM response in seconds
            llm_debug_logging: When True, logs full LLM requests and responses
            top_k: Number of chunks to retrieve for each query
        """
        # Use config default if model_path not specified
        if model_path is None:
            from ..config import default_config
            model_path = default_config.vosk_model_path
        
        # Store configuration parameters
        self._model_path = model_path
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_model_name = embedding_model
        self._lm_studio_url = lm_studio_url
        self._llm_timeout = llm_timeout
        self._llm_debug_logging = llm_debug_logging
        self.top_k = top_k
        
        # Initialize all component references to None (lazy loading)
        self._pdf_loader = None
        self._audio_transcriber = None
        self._video_processor = None
        self._text_chunker = None
        self._embedding_model = None
        self._vector_store = None
        self._llm = None
        
        logger.info("RAG Chatbot initialized with lazy loading")
    
    @property
    def vector_store(self):
        """
        Lazy property for vector store.
        Initializes VectorStore on first access and caches the instance.
        
        Returns:
            VectorStore: The initialized vector store instance
            
        Raises:
            RuntimeError: If vector store initialization fails
        """
        if self._vector_store is None:
            try:
                logger.info("Initializing vector store...")
                self._vector_store = VectorStore(
                    persist_directory=self._persist_directory,
                    collection_name=self._collection_name
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize vector store: {str(e)}"
                ) from e
        return self._vector_store
    
    @property
    def embedding_model(self):
        """
        Lazy property for embedding model.
        Initializes EmbeddingModel on first access and caches the instance.
        
        Returns:
            EmbeddingModel: The initialized embedding model instance
            
        Raises:
            RuntimeError: If embedding model initialization fails
        """
        if self._embedding_model is None:
            try:
                logger.info("Initializing embedding model...")
                self._embedding_model = EmbeddingModel(model_name=self._embedding_model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize embedding model: {str(e)}"
                ) from e
        return self._embedding_model
    
    @property
    def audio_transcriber(self):
        """
        Lazy property for audio transcriber.
        Initializes AudioTranscriber on first access and caches the instance.
        
        Returns:
            AudioTranscriber: The initialized audio transcriber instance
            
        Raises:
            RuntimeError: If audio transcriber initialization fails
        """
        if self._audio_transcriber is None:
            try:
                logger.info("Initializing audio transcriber...")
                self._audio_transcriber = AudioTranscriber(model_path=self._model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize audio transcriber: {str(e)}"
                ) from e
        return self._audio_transcriber
    
    @property
    def video_processor(self):
        """
        Lazy property for video processor.
        Initializes VideoProcessor on first access and caches the instance.
        
        Returns:
            VideoProcessor: The initialized video processor instance
            
        Raises:
            RuntimeError: If video processor initialization fails
        """
        if self._video_processor is None:
            try:
                logger.info("Initializing video processor...")
                self._video_processor = VideoProcessor(model_path=self._model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize video processor: {str(e)}"
                ) from e
        return self._video_processor
    
    @property
    def pdf_loader(self):
        """
        Lazy property for PDF loader.
        Initializes PDFLoader on first access and caches the instance.
        
        Returns:
            PDFLoader: The initialized PDF loader instance
            
        Raises:
            RuntimeError: If PDF loader initialization fails
        """
        if self._pdf_loader is None:
            try:
                logger.info("Initializing PDF loader...")
                self._pdf_loader = PDFLoader()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize PDF loader: {str(e)}"
                ) from e
        return self._pdf_loader
    
    @property
    def text_chunker(self):
        """
        Lazy property for text chunker.
        Initializes TextChunker on first access and caches the instance.
        
        Returns:
            TextChunker: The initialized text chunker instance
            
        Raises:
            RuntimeError: If text chunker initialization fails
        """
        if self._text_chunker is None:
            try:
                logger.info("Initializing text chunker...")
                self._text_chunker = TextChunker(
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                    model_name=f"sentence-transformers/{self._embedding_model_name}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize text chunker: {str(e)}"
                ) from e
        return self._text_chunker
    
    @property
    def llm(self):
        """
        Lazy property for LLM interface.
        Initializes LLMInterface on first access and caches the instance.
        
        Returns:
            LLMInterface: The initialized LLM interface instance
            
        Raises:
            RuntimeError: If LLM interface initialization fails
        """
        if self._llm is None:
            try:
                logger.info("Initializing LLM interface...")
                self._llm = LLMInterface(
                    base_url=self._lm_studio_url,
                    timeout=self._llm_timeout,
                    debug_logging=self._llm_debug_logging
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize LLM interface: {str(e)}"
                ) from e
        return self._llm
    
    def ingest_pdf(self, file_path: str) -> Dict[str, any]:
        """
        Ingest a PDF file into the knowledge base.
        
        Pipeline: load PDF → chunk → embed → store
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - chunks_added: Number of chunks added
                - source: Source file path
                - error: Error message if failed (optional)
                
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PDFProcessingError: If PDF processing fails
            EmbeddingError: If embedding generation fails
            StorageError: If storage fails
        """
        logger.info(f"Ingesting PDF: {file_path}")
        
        try:
            # Step 1: Load PDF
            text = self.pdf_loader.load(file_path)
            
            if not text or not text.strip():
                logger.warning(f"PDF contains no extractable text: {file_path}")
                return {
                    'success': False,
                    'chunks_added': 0,
                    'source': file_path,
                    'error': 'PDF contains no extractable text'
                }
            
            # Step 2: Chunk text
            chunks_with_metadata = self.text_chunker.chunk_with_metadata(
                text=text,
                source=str(Path(file_path).name)
            )
            
            if not chunks_with_metadata:
                logger.warning(f"No chunks created from PDF: {file_path}")
                return {
                    'success': False,
                    'chunks_added': 0,
                    'source': file_path,
                    'error': 'No chunks created from text'
                }
            
            # Extract texts and metadata
            texts = [chunk['text'] for chunk in chunks_with_metadata]
            metadata = [
                {
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index'],
                    'source_type': 'pdf'
                }
                for chunk in chunks_with_metadata
            ]
            
            # Step 3: Generate embeddings
            embeddings = self.embedding_model.embed_batch(texts)
            
            # Step 4: Store in vector database
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(f"Successfully ingested PDF: {file_path} ({len(texts)} chunks)")
            
            return {
                'success': True,
                'chunks_added': len(texts),
                'source': file_path
            }
            
        except (FileNotFoundError, PDFProcessingError, EmbeddingError, StorageError) as e:
            logger.error(f"Failed to ingest PDF {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error ingesting PDF {file_path}: {str(e)}")
            raise
    
    def ingest_audio(self, file_path: str) -> Dict[str, any]:
        """
        Ingest an audio file into the knowledge base.
        
        Pipeline: transcribe → chunk → embed → store
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - chunks_added: Number of chunks added
                - source: Source file path
                - error: Error message if failed (optional)
                
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            UnsupportedFormatError: If audio format is not supported
            TranscriptionError: If transcription fails
            EmbeddingError: If embedding generation fails
            StorageError: If storage fails
        """
        logger.info(f"Ingesting audio: {file_path}")
        
        try:
            # Step 1: Transcribe audio
            text = self.audio_transcriber.transcribe(file_path)
            
            if not text or not text.strip():
                logger.warning(f"Audio transcription produced no text: {file_path}")
                return {
                    'success': False,
                    'chunks_added': 0,
                    'source': file_path,
                    'error': 'Audio transcription produced no text'
                }
            
            # Step 2: Chunk text
            chunks_with_metadata = self.text_chunker.chunk_with_metadata(
                text=text,
                source=str(Path(file_path).name)
            )
            
            if not chunks_with_metadata:
                logger.warning(f"No chunks created from audio: {file_path}")
                return {
                    'success': False,
                    'chunks_added': 0,
                    'source': file_path,
                    'error': 'No chunks created from transcription'
                }
            
            # Extract texts and metadata
            texts = [chunk['text'] for chunk in chunks_with_metadata]
            metadata = [
                {
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index'],
                    'source_type': 'audio'
                }
                for chunk in chunks_with_metadata
            ]
            
            # Step 3: Generate embeddings
            embeddings = self.embedding_model.embed_batch(texts)
            
            # Step 4: Store in vector database
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(f"Successfully ingested audio: {file_path} ({len(texts)} chunks)")
            
            return {
                'success': True,
                'chunks_added': len(texts),
                'source': file_path
            }
            
        except (FileNotFoundError, UnsupportedFormatError, TranscriptionError, 
                EmbeddingError, StorageError) as e:
            logger.error(f"Failed to ingest audio {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error ingesting audio {file_path}: {str(e)}")
            raise
    
    def ingest_video(self, file_path: str) -> Dict[str, any]:
        """
        Ingest a video file into the knowledge base.
        
        Pipeline: extract audio → transcribe → chunk → embed → store
        
        Args:
            file_path: Path to the video file
            
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - chunks_added: Number of chunks added
                - source: Source file path
                - error: Error message if failed (optional)
                
        Raises:
            FileNotFoundError: If the video file doesn't exist
            UnsupportedFormatError: If video format is not supported
            VideoProcessingError: If video processing or transcription fails
            EmbeddingError: If embedding generation fails
            StorageError: If storage fails
        """
        logger.info(f"Ingesting video: {file_path}")
        
        try:
            # Step 1: Process video (extract audio and transcribe)
            text = self.video_processor.process_video(file_path)
            
            if not text or not text.strip():
                logger.warning(f"Video processing produced no text: {file_path}")
                return {
                    'success': False,
                    'chunks_added': 0,
                    'source': file_path,
                    'error': 'Video processing produced no text'
                }
            
            # Step 2: Chunk text
            chunks_with_metadata = self.text_chunker.chunk_with_metadata(
                text=text,
                source=str(Path(file_path).name)
            )
            
            if not chunks_with_metadata:
                logger.warning(f"No chunks created from video: {file_path}")
                return {
                    'success': False,
                    'chunks_added': 0,
                    'source': file_path,
                    'error': 'No chunks created from transcription'
                }
            
            # Extract texts and metadata
            texts = [chunk['text'] for chunk in chunks_with_metadata]
            metadata = [
                {
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index'],
                    'source_type': 'video'
                }
                for chunk in chunks_with_metadata
            ]
            
            # Step 3: Generate embeddings
            embeddings = self.embedding_model.embed_batch(texts)
            
            # Step 4: Store in vector database
            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadata=metadata
            )
            
            logger.info(f"Successfully ingested video: {file_path} ({len(texts)} chunks)")
            
            return {
                'success': True,
                'chunks_added': len(texts),
                'source': file_path
            }
            
        except (FileNotFoundError, UnsupportedFormatError, VideoProcessingError,
                EmbeddingError, StorageError) as e:
            logger.error(f"Failed to ingest video {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error ingesting video {file_path}: {str(e)}")
            raise
    
    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, any]:
        """
        Answer a question using the RAG pipeline.
        
        Pipeline: embed question → query database → generate answer
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve (uses default if not specified)
            
        Returns:
            Dictionary containing:
                - question: The original question
                - answer: The generated answer
                - context: List of retrieved context chunks
                - sources: List of source documents
                - error: Error message if failed (optional)
                
        Raises:
            EmbeddingError: If question embedding fails
            StorageError: If retrieval fails
            LLMError: If answer generation fails
        """
        if not question or not question.strip():
            return {
                'question': question,
                'answer': 'Please provide a valid question.',
                'context': [],
                'sources': [],
                'error': 'Empty question'
            }
        
        k = top_k if top_k is not None else self.top_k
        
        logger.info(f"Processing question: {question}")
        
        try:
            # Step 1: Embed the question
            question_embedding = self.embedding_model.embed(question)
            
            # Step 2: Query the vector database
            results = self.vector_store.query(
                query_embedding=question_embedding,
                top_k=k
            )
            
            # Check if we found any relevant context
            if not results:
                logger.warning("No relevant context found for question")
                return {
                    'question': question,
                    'answer': 'I could not find any relevant information in the knowledge base to answer your question.',
                    'context': [],
                    'sources': [],
                    'error': 'No relevant context found'
                }
            
            # Extract context texts and sources
            context_texts = [result['text'] for result in results]
            sources = [result['metadata'].get('source', 'Unknown') for result in results]
            
            # Step 3: Generate answer using LLM
            answer = self.llm.generate(
                question=question,
                context=context_texts
            )
            
            logger.info(f"Successfully answered question with {len(results)} context chunks")
            
            return {
                'question': question,
                'answer': answer,
                'context': context_texts,
                'sources': list(set(sources))  # Unique sources
            }
            
        except (EmbeddingError, StorageError, LLMError) as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error answering question: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing:
                - total_documents: Total number of chunks stored
                - collection_name: Name of the vector store collection
        """
        try:
            count = self.vector_store.count()
            return {
                'total_documents': count,
                'collection_name': self.vector_store.collection_name
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                'total_documents': 0,
                'collection_name': self.vector_store.collection_name,
                'error': str(e)
            }
    
    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        
        Raises:
            StorageError: If clear operation fails
        """
        logger.info("Clearing knowledge base...")
        self.vector_store.clear()
        logger.info("Knowledge base cleared successfully")
    
    def close(self) -> None:
        """
        Close all resources and connections.
        
        Important for persistent storage to avoid file locking issues.
        """
        logger.info("Closing RAG Chatbot...")
        self.vector_store.close()
        logger.info("RAG Chatbot closed")
