"""
Tests for LLM Interface.

Note: These tests verify the interface structure and error handling.
Full integration tests require LM Studio to be running and are in tests/manual/.
"""

import pytest
from src.rag.llm_interface import LLMInterface, LLMError


class TestLLMInterface:
    """Test suite for LLMInterface class."""
    
    def test_initialization_default(self):
        """Test LLM interface initialization with default parameters."""
        llm = LLMInterface()
        assert llm.base_url == "http://localhost:1234/v1"
        assert llm.model == "local-model"
        assert llm.client is not None
    
    def test_initialization_custom(self):
        """Test LLM interface initialization with custom parameters."""
        custom_url = "http://localhost:5000/v1"
        custom_model = "my-model"
        llm = LLMInterface(base_url=custom_url, model=custom_model)
        assert llm.base_url == custom_url
        assert llm.model == custom_model
    
    def test_generate_empty_question(self):
        """Test that empty question raises LLMError."""
        llm = LLMInterface()
        with pytest.raises(LLMError, match="Question cannot be empty"):
            llm.generate("", ["context"])
        
        with pytest.raises(LLMError, match="Question cannot be empty"):
            llm.generate("   ", ["context"])
    
    def test_generate_empty_context(self):
        """Test that empty context raises LLMError."""
        llm = LLMInterface()
        with pytest.raises(LLMError, match="Context cannot be empty"):
            llm.generate("What is this?", [])
    
    def test_format_context_single_chunk(self):
        """Test context formatting with a single chunk."""
        llm = LLMInterface()
        context = ["This is a test chunk."]
        formatted = llm._format_context(context)
        assert "[Context 1]" in formatted
        assert "This is a test chunk." in formatted
    
    def test_format_context_multiple_chunks(self):
        """Test context formatting with multiple chunks."""
        llm = LLMInterface()
        context = [
            "First chunk of text.",
            "Second chunk of text.",
            "Third chunk of text."
        ]
        formatted = llm._format_context(context)
        assert "[Context 1]" in formatted
        assert "[Context 2]" in formatted
        assert "[Context 3]" in formatted
        assert "First chunk of text." in formatted
        assert "Second chunk of text." in formatted
        assert "Third chunk of text." in formatted
    
    def test_create_prompt(self):
        """Test prompt creation."""
        llm = LLMInterface()
        question = "What is machine learning?"
        formatted_context = "[Context 1]\nMachine learning is a subset of AI."
        prompt = llm._create_prompt(question, formatted_context)
        
        assert "Context:" in prompt
        assert question in prompt
        assert formatted_context in prompt
        assert "Question:" in prompt
        assert "Answer based on the context" in prompt
    
    def test_format_context_preserves_order(self):
        """Test that context formatting preserves chunk order."""
        llm = LLMInterface()
        context = ["First", "Second", "Third", "Fourth"]
        formatted = llm._format_context(context)
        
        # Check that chunks appear in order
        first_pos = formatted.find("First")
        second_pos = formatted.find("Second")
        third_pos = formatted.find("Third")
        fourth_pos = formatted.find("Fourth")
        
        assert first_pos < second_pos < third_pos < fourth_pos


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
