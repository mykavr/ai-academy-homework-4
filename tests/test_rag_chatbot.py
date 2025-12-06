"""Unit tests for the RAGChatbot orchestration class."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.rag import RAGChatbot


class TestRAGChatbotLazyLoading:
    """Test suite for RAGChatbot lazy loading behavior."""
    
    def test_vector_store_initializes_on_first_access(self):
        """Test that vector_store initializes on first access."""
        with patch('src.rag.chatbot.VectorStore') as mock_store:
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            chatbot = RAGChatbot()
            
            # Verify vector_store is not initialized yet
            assert chatbot._vector_store is None
            
            # Access vector_store property
            result = chatbot.vector_store
            
            # Verify it was initialized
            mock_store.assert_called_once()
            assert result is mock_store_instance
            assert chatbot._vector_store is mock_store_instance
    
    def test_vector_store_caching(self):
        """Test that subsequent accesses return the same instance (caching)."""
        with patch('src.rag.chatbot.VectorStore') as mock_store:
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            chatbot = RAGChatbot()
            
            # Access vector_store multiple times
            first_access = chatbot.vector_store
            second_access = chatbot.vector_store
            third_access = chatbot.vector_store
            
            # Verify VectorStore was only initialized once
            mock_store.assert_called_once()
            
            # Verify all accesses return the same instance
            assert first_access is second_access
            assert second_access is third_access
            assert first_access is mock_store_instance
    
    def test_vector_store_initialization_error_wrapped(self):
        """Test that initialization errors are properly wrapped."""
        with patch('src.rag.chatbot.VectorStore') as mock_store:
            mock_store.side_effect = Exception("Database connection failed")
            
            chatbot = RAGChatbot()
            
            # Verify accessing vector_store raises RuntimeError with descriptive message
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.vector_store
            
            assert "Failed to initialize vector store" in str(exc_info.value)
            assert "Database connection failed" in str(exc_info.value)
    
    def test_vector_store_remains_none_when_not_accessed(self):
        """Test that vector_store remains None when not accessed."""
        with patch('src.rag.chatbot.VectorStore') as mock_store:
            chatbot = RAGChatbot()
            
            # Verify vector_store is None and VectorStore was not called
            assert chatbot._vector_store is None
            mock_store.assert_not_called()
    
    def test_embedding_model_initializes_on_first_access(self):
        """Test that embedding_model initializes on first access."""
        with patch('src.rag.chatbot.EmbeddingModel') as mock_embedding:
            mock_embedding_instance = Mock()
            mock_embedding.return_value = mock_embedding_instance
            
            chatbot = RAGChatbot(embedding_model="test-model")
            
            # Verify embedding_model is not initialized yet
            assert chatbot._embedding_model is None
            
            # Access embedding_model property
            result = chatbot.embedding_model
            
            # Verify it was initialized with correct model name
            mock_embedding.assert_called_once_with(model_name="test-model")
            assert result is mock_embedding_instance
            assert chatbot._embedding_model is mock_embedding_instance
    
    def test_embedding_model_caching(self):
        """Test that subsequent accesses return the same instance (caching)."""
        with patch('src.rag.chatbot.EmbeddingModel') as mock_embedding:
            mock_embedding_instance = Mock()
            mock_embedding.return_value = mock_embedding_instance
            
            chatbot = RAGChatbot()
            
            # Access embedding_model multiple times
            first_access = chatbot.embedding_model
            second_access = chatbot.embedding_model
            third_access = chatbot.embedding_model
            
            # Verify EmbeddingModel was only initialized once
            mock_embedding.assert_called_once()
            
            # Verify all accesses return the same instance
            assert first_access is second_access
            assert second_access is third_access
            assert first_access is mock_embedding_instance
    
    def test_embedding_model_initialization_error_wrapped(self):
        """Test that initialization errors are properly wrapped."""
        with patch('src.rag.chatbot.EmbeddingModel') as mock_embedding:
            mock_embedding.side_effect = Exception("Model download failed")
            
            chatbot = RAGChatbot()
            
            # Verify accessing embedding_model raises RuntimeError with descriptive message
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.embedding_model
            
            assert "Failed to initialize embedding model" in str(exc_info.value)
            assert "Model download failed" in str(exc_info.value)
    
    def test_embedding_model_remains_none_when_not_accessed(self):
        """Test that embedding_model remains None when not accessed."""
        with patch('src.rag.chatbot.EmbeddingModel') as mock_embedding:
            chatbot = RAGChatbot()
            
            # Verify embedding_model is None and EmbeddingModel was not called
            assert chatbot._embedding_model is None
            mock_embedding.assert_not_called()
    
    def test_audio_transcriber_initializes_on_first_access(self):
        """Test that audio_transcriber initializes on first access."""
        with patch('src.rag.chatbot.AudioTranscriber') as mock_audio:
            mock_audio_instance = Mock()
            mock_audio.return_value = mock_audio_instance
            
            chatbot = RAGChatbot(model_path="/path/to/model")
            
            # Verify audio_transcriber is not initialized yet
            assert chatbot._audio_transcriber is None
            
            # Access audio_transcriber property
            result = chatbot.audio_transcriber
            
            # Verify it was initialized with correct model path
            mock_audio.assert_called_once_with(model_path="/path/to/model")
            assert result is mock_audio_instance
            assert chatbot._audio_transcriber is mock_audio_instance
    
    def test_audio_transcriber_caching(self):
        """Test that subsequent accesses return the same instance (caching)."""
        with patch('src.rag.chatbot.AudioTranscriber') as mock_audio:
            mock_audio_instance = Mock()
            mock_audio.return_value = mock_audio_instance
            
            chatbot = RAGChatbot()
            
            # Access audio_transcriber multiple times
            first_access = chatbot.audio_transcriber
            second_access = chatbot.audio_transcriber
            third_access = chatbot.audio_transcriber
            
            # Verify AudioTranscriber was only initialized once
            mock_audio.assert_called_once()
            
            # Verify all accesses return the same instance
            assert first_access is second_access
            assert second_access is third_access
            assert first_access is mock_audio_instance
    
    def test_audio_transcriber_initialization_error_wrapped(self):
        """Test that initialization errors are properly wrapped."""
        with patch('src.rag.chatbot.AudioTranscriber') as mock_audio:
            mock_audio.side_effect = Exception("Vosk model not found")
            
            chatbot = RAGChatbot()
            
            # Verify accessing audio_transcriber raises RuntimeError with descriptive message
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.audio_transcriber
            
            assert "Failed to initialize audio transcriber" in str(exc_info.value)
            assert "Vosk model not found" in str(exc_info.value)
    
    def test_audio_transcriber_remains_none_when_not_accessed(self):
        """Test that audio_transcriber remains None when not accessed."""
        with patch('src.rag.chatbot.AudioTranscriber') as mock_audio:
            chatbot = RAGChatbot()
            
            # Verify audio_transcriber is None and AudioTranscriber was not called
            assert chatbot._audio_transcriber is None
            mock_audio.assert_not_called()
    
    def test_video_processor_initializes_on_first_access(self):
        """Test that video_processor initializes on first access."""
        with patch('src.rag.chatbot.VideoProcessor') as mock_video:
            mock_video_instance = Mock()
            mock_video.return_value = mock_video_instance
            
            chatbot = RAGChatbot(model_path="/path/to/model")
            
            # Verify video_processor is not initialized yet
            assert chatbot._video_processor is None
            
            # Access video_processor property
            result = chatbot.video_processor
            
            # Verify it was initialized with correct model path
            mock_video.assert_called_once_with(model_path="/path/to/model")
            assert result is mock_video_instance
            assert chatbot._video_processor is mock_video_instance
    
    def test_video_processor_caching(self):
        """Test that subsequent accesses return the same instance (caching)."""
        with patch('src.rag.chatbot.VideoProcessor') as mock_video:
            mock_video_instance = Mock()
            mock_video.return_value = mock_video_instance
            
            chatbot = RAGChatbot()
            
            # Access video_processor multiple times
            first_access = chatbot.video_processor
            second_access = chatbot.video_processor
            third_access = chatbot.video_processor
            
            # Verify VideoProcessor was only initialized once
            mock_video.assert_called_once()
            
            # Verify all accesses return the same instance
            assert first_access is second_access
            assert second_access is third_access
            assert first_access is mock_video_instance
    
    def test_video_processor_initialization_error_wrapped(self):
        """Test that initialization errors are properly wrapped."""
        with patch('src.rag.chatbot.VideoProcessor') as mock_video:
            mock_video.side_effect = Exception("FFmpeg not found")
            
            chatbot = RAGChatbot()
            
            # Verify accessing video_processor raises RuntimeError with descriptive message
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.video_processor
            
            assert "Failed to initialize video processor" in str(exc_info.value)
            assert "FFmpeg not found" in str(exc_info.value)
    
    def test_video_processor_remains_none_when_not_accessed(self):
        """Test that video_processor remains None when not accessed."""
        with patch('src.rag.chatbot.VideoProcessor') as mock_video:
            chatbot = RAGChatbot()
            
            # Verify video_processor is None and VideoProcessor was not called
            assert chatbot._video_processor is None
            mock_video.assert_not_called()
    
    def test_pdf_loader_initializes_on_first_access_and_caches(self):
        """Test that pdf_loader initializes on first access and caches."""
        with patch('src.rag.chatbot.PDFLoader') as mock_pdf:
            mock_pdf_instance = Mock()
            mock_pdf.return_value = mock_pdf_instance
            
            chatbot = RAGChatbot()
            
            # Verify pdf_loader is not initialized yet
            assert chatbot._pdf_loader is None
            
            # Access pdf_loader multiple times
            first_access = chatbot.pdf_loader
            second_access = chatbot.pdf_loader
            
            # Verify it was initialized only once
            mock_pdf.assert_called_once()
            assert first_access is second_access
            assert first_access is mock_pdf_instance
    
    def test_text_chunker_initializes_on_first_access_and_caches(self):
        """Test that text_chunker initializes on first access and caches."""
        with patch('src.rag.chatbot.TextChunker') as mock_chunker:
            mock_chunker_instance = Mock()
            mock_chunker.return_value = mock_chunker_instance
            
            chatbot = RAGChatbot(chunk_size=256, chunk_overlap=50, embedding_model="test-model")
            
            # Verify text_chunker is not initialized yet
            assert chatbot._text_chunker is None
            
            # Access text_chunker multiple times
            first_access = chatbot.text_chunker
            second_access = chatbot.text_chunker
            
            # Verify it was initialized only once with correct parameters
            mock_chunker.assert_called_once_with(
                chunk_size=256,
                chunk_overlap=50,
                model_name="sentence-transformers/test-model"
            )
            assert first_access is second_access
            assert first_access is mock_chunker_instance
    
    def test_llm_initializes_on_first_access_and_caches(self):
        """Test that llm initializes on first access and caches."""
        with patch('src.rag.chatbot.LLMInterface') as mock_llm:
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            chatbot = RAGChatbot(lm_studio_url="http://test:8080", llm_timeout=30, llm_debug_logging=True)
            
            # Verify llm is not initialized yet
            assert chatbot._llm is None
            
            # Access llm multiple times
            first_access = chatbot.llm
            second_access = chatbot.llm
            
            # Verify it was initialized only once with correct parameters
            mock_llm.assert_called_once_with(
                base_url="http://test:8080",
                timeout=30,
                debug_logging=True
            )
            assert first_access is second_access
            assert first_access is mock_llm_instance
    
    def test_pdf_loader_initialization_error_wrapped(self):
        """Test error handling for pdf_loader."""
        with patch('src.rag.chatbot.PDFLoader') as mock_pdf:
            mock_pdf.side_effect = Exception("PDF library error")
            
            chatbot = RAGChatbot()
            
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.pdf_loader
            
            assert "Failed to initialize PDF loader" in str(exc_info.value)
            assert "PDF library error" in str(exc_info.value)
    
    def test_text_chunker_initialization_error_wrapped(self):
        """Test error handling for text_chunker."""
        with patch('src.rag.chatbot.TextChunker') as mock_chunker:
            mock_chunker.side_effect = Exception("Tokenizer error")
            
            chatbot = RAGChatbot()
            
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.text_chunker
            
            assert "Failed to initialize text chunker" in str(exc_info.value)
            assert "Tokenizer error" in str(exc_info.value)
    
    def test_llm_initialization_error_wrapped(self):
        """Test error handling for llm."""
        with patch('src.rag.chatbot.LLMInterface') as mock_llm:
            mock_llm.side_effect = Exception("Connection refused")
            
            chatbot = RAGChatbot()
            
            with pytest.raises(RuntimeError) as exc_info:
                _ = chatbot.llm
            
            assert "Failed to initialize LLM interface" in str(exc_info.value)
            assert "Connection refused" in str(exc_info.value)


class TestRAGChatbot:
    """Test suite for RAGChatbot orchestration."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the test database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_chatbot_components(self):
        """Mock all the components to test orchestration without dependencies."""
        with patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Configure mocks
            mock_pdf_instance = Mock()
            mock_pdf.return_value = mock_pdf_instance
            
            mock_audio_instance = Mock()
            mock_audio.return_value = mock_audio_instance
            
            mock_video_instance = Mock()
            mock_video.return_value = mock_video_instance
            
            mock_chunker_instance = Mock()
            mock_chunker.return_value = mock_chunker_instance
            
            mock_embedding_instance = Mock()
            mock_embedding.return_value = mock_embedding_instance
            
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            yield {
                'pdf_loader': mock_pdf_instance,
                'audio_transcriber': mock_audio_instance,
                'video_processor': mock_video_instance,
                'text_chunker': mock_chunker_instance,
                'embedding_model': mock_embedding_instance,
                'vector_store': mock_store_instance,
                'llm': mock_llm_instance
            }
    
    def test_initialization_creates_all_components(self, mock_chatbot_components):
        """Test that RAGChatbot initializes all required components."""
        chatbot = RAGChatbot()
        
        # Verify all components are initialized
        assert chatbot.pdf_loader is not None
        assert chatbot.audio_transcriber is not None
        assert chatbot.video_processor is not None
        assert chatbot.text_chunker is not None
        assert chatbot.embedding_model is not None
        assert chatbot.vector_store is not None
        assert chatbot.llm is not None
    
    def test_ingest_pdf_pipeline(self, mock_chatbot_components):
        """Test that ingest_pdf orchestrates the correct pipeline."""
        # Setup mocks
        mock_chatbot_components['pdf_loader'].load.return_value = "Sample PDF text content."
        mock_chatbot_components['text_chunker'].chunk_with_metadata.return_value = [
            {'text': 'Chunk 1', 'source': 'test.pdf', 'chunk_index': 0},
            {'text': 'Chunk 2', 'source': 'test.pdf', 'chunk_index': 1}
        ]
        mock_chatbot_components['embedding_model'].embed_batch.return_value = [
            [0.1] * 768,
            [0.2] * 768
        ]
        
        # Create chatbot and ingest PDF
        chatbot = RAGChatbot()
        result = chatbot.ingest_pdf("test.pdf")
        
        # Verify pipeline steps were called
        mock_chatbot_components['pdf_loader'].load.assert_called_once_with("test.pdf")
        mock_chatbot_components['text_chunker'].chunk_with_metadata.assert_called_once()
        mock_chatbot_components['embedding_model'].embed_batch.assert_called_once()
        mock_chatbot_components['vector_store'].add_documents.assert_called_once()
        
        # Verify result
        assert result['success'] is True
        assert result['chunks_added'] == 2
        assert result['source'] == "test.pdf"
    
    def test_ingest_audio_pipeline(self, mock_chatbot_components):
        """Test that ingest_audio orchestrates the correct pipeline."""
        # Setup mocks
        mock_chatbot_components['audio_transcriber'].transcribe.return_value = "Sample audio transcription."
        mock_chatbot_components['text_chunker'].chunk_with_metadata.return_value = [
            {'text': 'Chunk 1', 'source': 'test.mp3', 'chunk_index': 0}
        ]
        mock_chatbot_components['embedding_model'].embed_batch.return_value = [
            [0.1] * 768
        ]
        
        # Create chatbot and ingest audio
        chatbot = RAGChatbot()
        result = chatbot.ingest_audio("test.mp3")
        
        # Verify pipeline steps were called
        mock_chatbot_components['audio_transcriber'].transcribe.assert_called_once_with("test.mp3")
        mock_chatbot_components['text_chunker'].chunk_with_metadata.assert_called_once()
        mock_chatbot_components['embedding_model'].embed_batch.assert_called_once()
        mock_chatbot_components['vector_store'].add_documents.assert_called_once()
        
        # Verify result
        assert result['success'] is True
        assert result['chunks_added'] == 1
        assert result['source'] == "test.mp3"
    
    def test_ingest_video_pipeline(self, mock_chatbot_components):
        """Test that ingest_video orchestrates the correct pipeline."""
        # Setup mocks
        mock_chatbot_components['video_processor'].process_video.return_value = "Sample video transcription."
        mock_chatbot_components['text_chunker'].chunk_with_metadata.return_value = [
            {'text': 'Chunk 1', 'source': 'test.mp4', 'chunk_index': 0}
        ]
        mock_chatbot_components['embedding_model'].embed_batch.return_value = [
            [0.1] * 768
        ]
        
        # Create chatbot and ingest video
        chatbot = RAGChatbot()
        result = chatbot.ingest_video("test.mp4")
        
        # Verify pipeline steps were called
        mock_chatbot_components['video_processor'].process_video.assert_called_once_with("test.mp4")
        mock_chatbot_components['text_chunker'].chunk_with_metadata.assert_called_once()
        mock_chatbot_components['embedding_model'].embed_batch.assert_called_once()
        mock_chatbot_components['vector_store'].add_documents.assert_called_once()
        
        # Verify result
        assert result['success'] is True
        assert result['chunks_added'] == 1
        assert result['source'] == "test.mp4"
    
    def test_ask_pipeline(self, mock_chatbot_components):
        """Test that ask orchestrates the correct pipeline."""
        # Setup mocks
        mock_chatbot_components['embedding_model'].embed.return_value = [0.1] * 768
        mock_chatbot_components['vector_store'].query.return_value = [
            {
                'text': 'Context chunk 1',
                'metadata': {'source': 'test.pdf', 'chunk_index': 0},
                'distance': 0.1
            },
            {
                'text': 'Context chunk 2',
                'metadata': {'source': 'test.pdf', 'chunk_index': 1},
                'distance': 0.2
            }
        ]
        mock_chatbot_components['llm'].generate.return_value = "This is the answer."
        
        # Create chatbot and ask question
        chatbot = RAGChatbot()
        result = chatbot.ask("What is machine learning?")
        
        # Verify pipeline steps were called
        mock_chatbot_components['embedding_model'].embed.assert_called_once_with("What is machine learning?")
        mock_chatbot_components['vector_store'].query.assert_called_once()
        mock_chatbot_components['llm'].generate.assert_called_once()
        
        # Verify result
        assert result['question'] == "What is machine learning?"
        assert result['answer'] == "This is the answer."
        assert len(result['context']) == 2
        assert 'test.pdf' in result['sources']
    
    def test_ask_with_empty_question(self, mock_chatbot_components):
        """Test that ask handles empty questions gracefully."""
        chatbot = RAGChatbot()
        result = chatbot.ask("")
        
        # Verify error handling
        assert 'error' in result
        assert result['answer'] == 'Please provide a valid question.'
        assert result['context'] == []
        assert result['sources'] == []
    
    def test_ask_with_no_relevant_context(self, mock_chatbot_components):
        """Test that ask handles no relevant context gracefully."""
        # Setup mocks
        mock_chatbot_components['embedding_model'].embed.return_value = [0.1] * 768
        mock_chatbot_components['vector_store'].query.return_value = []
        
        chatbot = RAGChatbot()
        result = chatbot.ask("What is quantum computing?")
        
        # Verify error handling
        assert 'error' in result
        assert 'could not find any relevant information' in result['answer'].lower()
        assert result['context'] == []
        assert result['sources'] == []
    
    def test_ingest_pdf_with_empty_text(self, mock_chatbot_components):
        """Test that ingest_pdf handles empty text gracefully."""
        # Setup mocks
        mock_chatbot_components['pdf_loader'].load.return_value = ""
        
        chatbot = RAGChatbot()
        result = chatbot.ingest_pdf("empty.pdf")
        
        # Verify error handling
        assert result['success'] is False
        assert result['chunks_added'] == 0
        assert 'error' in result
    
    def test_ingest_audio_with_empty_transcription(self, mock_chatbot_components):
        """Test that ingest_audio handles empty transcription gracefully."""
        # Setup mocks
        mock_chatbot_components['audio_transcriber'].transcribe.return_value = ""
        
        chatbot = RAGChatbot()
        result = chatbot.ingest_audio("silent.mp3")
        
        # Verify error handling
        assert result['success'] is False
        assert result['chunks_added'] == 0
        assert 'error' in result
    
    def test_ingest_video_with_empty_transcription(self, mock_chatbot_components):
        """Test that ingest_video handles empty transcription gracefully."""
        # Setup mocks
        mock_chatbot_components['video_processor'].process_video.return_value = ""
        
        chatbot = RAGChatbot()
        result = chatbot.ingest_video("silent.mp4")
        
        # Verify error handling
        assert result['success'] is False
        assert result['chunks_added'] == 0
        assert 'error' in result
    
    def test_metadata_includes_source_type(self, mock_chatbot_components):
        """Test that metadata includes correct source_type for each ingestion method."""
        # Setup mocks for PDF
        mock_chatbot_components['pdf_loader'].load.return_value = "PDF content"
        mock_chatbot_components['text_chunker'].chunk_with_metadata.return_value = [
            {'text': 'Chunk', 'source': 'test.pdf', 'chunk_index': 0}
        ]
        mock_chatbot_components['embedding_model'].embed_batch.return_value = [[0.1] * 768]
        
        chatbot = RAGChatbot()
        chatbot.ingest_pdf("test.pdf")
        
        # Verify metadata includes source_type='pdf'
        call_args = mock_chatbot_components['vector_store'].add_documents.call_args
        metadata = call_args[1]['metadata']
        assert metadata[0]['source_type'] == 'pdf'
        
        # Setup mocks for audio
        mock_chatbot_components['audio_transcriber'].transcribe.return_value = "Audio content"
        mock_chatbot_components['text_chunker'].chunk_with_metadata.return_value = [
            {'text': 'Chunk', 'source': 'test.mp3', 'chunk_index': 0}
        ]
        
        chatbot.ingest_audio("test.mp3")
        
        # Verify metadata includes source_type='audio'
        call_args = mock_chatbot_components['vector_store'].add_documents.call_args
        metadata = call_args[1]['metadata']
        assert metadata[0]['source_type'] == 'audio'
        
        # Setup mocks for video
        mock_chatbot_components['video_processor'].process_video.return_value = "Video content"
        mock_chatbot_components['text_chunker'].chunk_with_metadata.return_value = [
            {'text': 'Chunk', 'source': 'test.mp4', 'chunk_index': 0}
        ]
        
        chatbot.ingest_video("test.mp4")
        
        # Verify metadata includes source_type='video'
        call_args = mock_chatbot_components['vector_store'].add_documents.call_args
        metadata = call_args[1]['metadata']
        assert metadata[0]['source_type'] == 'video'
    
    def test_get_stats(self, mock_chatbot_components):
        """Test that get_stats returns correct information."""
        mock_chatbot_components['vector_store'].count.return_value = 42
        mock_chatbot_components['vector_store'].collection_name = "test_collection"
        
        chatbot = RAGChatbot()
        stats = chatbot.get_stats()
        
        assert stats['total_documents'] == 42
        assert stats['collection_name'] == "test_collection"
    
    def test_clear_knowledge_base(self, mock_chatbot_components):
        """Test that clear_knowledge_base calls vector store clear."""
        chatbot = RAGChatbot()
        chatbot.clear_knowledge_base()
        
        mock_chatbot_components['vector_store'].clear.assert_called_once()
    
    def test_close(self, mock_chatbot_components):
        """Test that close calls vector store close."""
        chatbot = RAGChatbot()
        chatbot.close()
        
        mock_chatbot_components['vector_store'].close.assert_called_once()


class TestCommandSpecificComponentLoading:
    """Test suite for verifying that commands load only their required components."""
    
    def test_stats_command_loads_only_vector_store(self):
        """Test that stats command loads only vector_store."""
        with patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Setup vector store mock
            mock_store_instance = Mock()
            mock_store_instance.count.return_value = 10
            mock_store_instance.collection_name = "test_collection"
            mock_store.return_value = mock_store_instance
            
            # Create chatbot and call get_stats
            chatbot = RAGChatbot()
            stats = chatbot.get_stats()
            
            # Verify only vector_store was initialized
            mock_store.assert_called_once()
            
            # Verify other components were NOT initialized
            mock_embedding.assert_not_called()
            mock_audio.assert_not_called()
            mock_video.assert_not_called()
            mock_pdf.assert_not_called()
            mock_chunker.assert_not_called()
            mock_llm.assert_not_called()
            
            # Verify stats are correct
            assert stats['total_documents'] == 10
            assert stats['collection_name'] == "test_collection"
    
    def test_clear_command_loads_only_vector_store(self):
        """Test that clear command loads only vector_store."""
        with patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Setup vector store mock
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            # Create chatbot and call clear_knowledge_base
            chatbot = RAGChatbot()
            chatbot.clear_knowledge_base()
            
            # Verify only vector_store was initialized
            mock_store.assert_called_once()
            mock_store_instance.clear.assert_called_once()
            
            # Verify other components were NOT initialized
            mock_embedding.assert_not_called()
            mock_audio.assert_not_called()
            mock_video.assert_not_called()
            mock_pdf.assert_not_called()
            mock_chunker.assert_not_called()
            mock_llm.assert_not_called()
    
    def test_ask_command_loads_embedding_vector_store_and_llm(self):
        """Test that ask command loads embedding_model, vector_store, and llm."""
        with patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Setup mocks
            mock_store_instance = Mock()
            mock_store_instance.query.return_value = [
                {
                    'text': 'Context chunk',
                    'metadata': {'source': 'test.pdf', 'chunk_index': 0},
                    'distance': 0.1
                }
            ]
            mock_store.return_value = mock_store_instance
            
            mock_embedding_instance = Mock()
            mock_embedding_instance.embed.return_value = [0.1] * 768
            mock_embedding.return_value = mock_embedding_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.generate.return_value = "Answer"
            mock_llm.return_value = mock_llm_instance
            
            # Create chatbot and ask question
            chatbot = RAGChatbot()
            result = chatbot.ask("What is AI?")
            
            # Verify required components were initialized
            mock_embedding.assert_called_once()
            mock_store.assert_called_once()
            mock_llm.assert_called_once()
            
            # Verify transcription components were NOT initialized
            mock_audio.assert_not_called()
            mock_video.assert_not_called()
            
            # Verify other components were NOT initialized
            mock_pdf.assert_not_called()
            mock_chunker.assert_not_called()
            
            # Verify result
            assert result['answer'] == "Answer"
    
    def test_ingest_pdf_loads_pdf_chunker_embedding_vector_store(self):
        """Test that ingest_pdf loads pdf_loader, text_chunker, embedding_model, vector_store."""
        with patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Setup mocks
            mock_pdf_instance = Mock()
            mock_pdf_instance.load.return_value = "PDF content"
            mock_pdf.return_value = mock_pdf_instance
            
            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk_with_metadata.return_value = [
                {'text': 'Chunk', 'source': 'test.pdf', 'chunk_index': 0}
            ]
            mock_chunker.return_value = mock_chunker_instance
            
            mock_embedding_instance = Mock()
            mock_embedding_instance.embed_batch.return_value = [[0.1] * 768]
            mock_embedding.return_value = mock_embedding_instance
            
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            # Create chatbot and ingest PDF
            chatbot = RAGChatbot()
            result = chatbot.ingest_pdf("test.pdf")
            
            # Verify required components were initialized
            mock_pdf.assert_called_once()
            mock_chunker.assert_called_once()
            mock_embedding.assert_called_once()
            mock_store.assert_called_once()
            
            # Verify transcription components were NOT initialized
            mock_audio.assert_not_called()
            mock_video.assert_not_called()
            
            # Verify LLM was NOT initialized
            mock_llm.assert_not_called()
            
            # Verify result
            assert result['success'] is True
    
    def test_ingest_audio_loads_audio_chunker_embedding_vector_store(self):
        """Test that ingest_audio loads audio_transcriber, text_chunker, embedding_model, vector_store."""
        with patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Setup mocks
            mock_audio_instance = Mock()
            mock_audio_instance.transcribe.return_value = "Audio transcription"
            mock_audio.return_value = mock_audio_instance
            
            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk_with_metadata.return_value = [
                {'text': 'Chunk', 'source': 'test.mp3', 'chunk_index': 0}
            ]
            mock_chunker.return_value = mock_chunker_instance
            
            mock_embedding_instance = Mock()
            mock_embedding_instance.embed_batch.return_value = [[0.1] * 768]
            mock_embedding.return_value = mock_embedding_instance
            
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            # Create chatbot and ingest audio
            chatbot = RAGChatbot()
            result = chatbot.ingest_audio("test.mp3")
            
            # Verify required components were initialized
            mock_audio.assert_called_once()
            mock_chunker.assert_called_once()
            mock_embedding.assert_called_once()
            mock_store.assert_called_once()
            
            # Verify other transcription components were NOT initialized
            mock_video.assert_not_called()
            mock_pdf.assert_not_called()
            
            # Verify LLM was NOT initialized
            mock_llm.assert_not_called()
            
            # Verify result
            assert result['success'] is True
    
    def test_ingest_video_loads_video_chunker_embedding_vector_store(self):
        """Test that ingest_video loads video_processor, text_chunker, embedding_model, vector_store."""
        with patch('src.rag.chatbot.VectorStore') as mock_store, \
             patch('src.rag.chatbot.EmbeddingModel') as mock_embedding, \
             patch('src.rag.chatbot.AudioTranscriber') as mock_audio, \
             patch('src.rag.chatbot.VideoProcessor') as mock_video, \
             patch('src.rag.chatbot.PDFLoader') as mock_pdf, \
             patch('src.rag.chatbot.TextChunker') as mock_chunker, \
             patch('src.rag.chatbot.LLMInterface') as mock_llm:
            
            # Setup mocks
            mock_video_instance = Mock()
            mock_video_instance.process_video.return_value = "Video transcription"
            mock_video.return_value = mock_video_instance
            
            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk_with_metadata.return_value = [
                {'text': 'Chunk', 'source': 'test.mp4', 'chunk_index': 0}
            ]
            mock_chunker.return_value = mock_chunker_instance
            
            mock_embedding_instance = Mock()
            mock_embedding_instance.embed_batch.return_value = [[0.1] * 768]
            mock_embedding.return_value = mock_embedding_instance
            
            mock_store_instance = Mock()
            mock_store.return_value = mock_store_instance
            
            # Create chatbot and ingest video
            chatbot = RAGChatbot()
            result = chatbot.ingest_video("test.mp4")
            
            # Verify required components were initialized
            mock_video.assert_called_once()
            mock_chunker.assert_called_once()
            mock_embedding.assert_called_once()
            mock_store.assert_called_once()
            
            # Verify other transcription components were NOT initialized
            mock_audio.assert_not_called()
            mock_pdf.assert_not_called()
            
            # Verify LLM was NOT initialized
            mock_llm.assert_not_called()
            
            # Verify result
            assert result['success'] is True
