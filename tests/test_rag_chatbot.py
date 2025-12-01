"""Unit tests for the RAGChatbot orchestration class."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.rag import RAGChatbot


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
