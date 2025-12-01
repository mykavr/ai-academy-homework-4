"""
Manual test for RAG Chatbot orchestration.

This test verifies that the RAGChatbot class correctly orchestrates all components.
Run this manually to test the complete pipeline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag.chatbot import RAGChatbot
import tempfile
import shutil


def test_rag_chatbot_initialization():
    """Test that RAGChatbot initializes successfully."""
    print("Testing RAGChatbot initialization...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        chatbot = RAGChatbot(persist_directory=temp_dir)
        
        # Verify all components are initialized
        assert chatbot.pdf_loader is not None, "PDF loader not initialized"
        assert chatbot.audio_transcriber is not None, "Audio transcriber not initialized"
        assert chatbot.video_processor is not None, "Video processor not initialized"
        assert chatbot.text_chunker is not None, "Text chunker not initialized"
        assert chatbot.embedding_model is not None, "Embedding model not initialized"
        assert chatbot.vector_store is not None, "Vector store not initialized"
        assert chatbot.llm is not None, "LLM interface not initialized"
        
        print("✓ All components initialized successfully")
        
        # Test get_stats
        stats = chatbot.get_stats()
        print(f"✓ Stats: {stats}")
        assert stats['total_documents'] == 0, "Expected empty database"
        
        # Close chatbot
        chatbot.close()
        print("✓ Chatbot closed successfully")
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("\n✅ RAGChatbot initialization test passed!")


def test_rag_chatbot_methods_exist():
    """Test that all required methods exist."""
    print("\nTesting RAGChatbot methods...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        chatbot = RAGChatbot(persist_directory=temp_dir)
        
        # Check all required methods exist
        assert hasattr(chatbot, 'ingest_pdf'), "Missing ingest_pdf method"
        assert hasattr(chatbot, 'ingest_audio'), "Missing ingest_audio method"
        assert hasattr(chatbot, 'ingest_video'), "Missing ingest_video method"
        assert hasattr(chatbot, 'ask'), "Missing ask method"
        assert hasattr(chatbot, 'get_stats'), "Missing get_stats method"
        assert hasattr(chatbot, 'clear_knowledge_base'), "Missing clear_knowledge_base method"
        assert hasattr(chatbot, 'close'), "Missing close method"
        
        print("✓ All required methods exist")
        
        # Test that methods are callable
        assert callable(chatbot.ingest_pdf), "ingest_pdf not callable"
        assert callable(chatbot.ingest_audio), "ingest_audio not callable"
        assert callable(chatbot.ingest_video), "ingest_video not callable"
        assert callable(chatbot.ask), "ask not callable"
        assert callable(chatbot.get_stats), "get_stats not callable"
        assert callable(chatbot.clear_knowledge_base), "clear_knowledge_base not callable"
        assert callable(chatbot.close), "close not callable"
        
        print("✓ All methods are callable")
        
        chatbot.close()
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("\n✅ RAGChatbot methods test passed!")


def test_ask_with_empty_question():
    """Test that ask handles empty questions gracefully."""
    print("\nTesting ask with empty question...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        chatbot = RAGChatbot(persist_directory=temp_dir)
        
        # Test empty question
        result = chatbot.ask("")
        assert 'error' in result, "Expected error in result"
        assert result['answer'] == 'Please provide a valid question.', "Unexpected answer"
        assert result['context'] == [], "Expected empty context"
        assert result['sources'] == [], "Expected empty sources"
        
        print("✓ Empty question handled correctly")
        
        # Test whitespace-only question
        result = chatbot.ask("   ")
        assert 'error' in result, "Expected error in result"
        
        print("✓ Whitespace question handled correctly")
        
        chatbot.close()
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("\n✅ Empty question test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Chatbot Manual Tests")
    print("=" * 60)
    
    try:
        test_rag_chatbot_initialization()
        test_rag_chatbot_methods_exist()
        test_ask_with_empty_question()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
