"""
Manual integration tests for LLM Interface.

These tests require LM Studio to be running with a model loaded.

Setup:
1. Install and open LM Studio
2. Load a model (e.g., Llama 2 7B, Mistral 7B, or similar)
3. Start the local server (default: http://localhost:1234)
4. Run this test file from the project root:
   python tests/manual/test_llm_manual.py

Or from the tests/manual directory:
   python -m tests.manual.test_llm_manual
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.llm_interface import LLMInterface, LLMError


def test_llm_connection():
    """Test basic connection to LM Studio."""
    print("\n=== Testing LM Studio Connection ===")
    
    try:
        llm = LLMInterface()
        print("✓ Successfully initialized LLM interface")
        print(f"  Base URL: {llm.base_url}")
        print(f"  Model: {llm.model}")
        return True
    except LLMError as e:
        print(f"✗ Failed to initialize LLM interface: {e}")
        return False


def test_simple_generation():
    """Test simple answer generation."""
    print("\n=== Testing Simple Generation ===")
    
    try:
        llm = LLMInterface()
        
        question = "What is machine learning?"
        context = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "It involves training algorithms on datasets to make predictions or decisions without being explicitly programmed."
        ]
        
        print(f"Question: {question}")
        print(f"Context chunks: {len(context)}")
        print("Generating answer (this may take a few seconds)...")
        
        answer = llm.generate(question, context)
        
        print(f"\nGenerated Answer:\n{answer}")
        print("\n✓ Successfully generated answer")
        return True
        
    except LLMError as e:
        print(f"✗ Failed to generate answer: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_lecture_content_qa():
    """Test Q&A with lecture-style content."""
    print("\n=== Testing Lecture Content Q&A ===")
    
    try:
        llm = LLMInterface()
        
        # Simulate lecture content chunks
        context = [
            "In today's lecture, we discussed neural networks. A neural network is composed of layers of interconnected nodes.",
            "Each node applies a mathematical function to its inputs and passes the result to the next layer.",
            "Training a neural network involves adjusting the weights of connections to minimize prediction error."
        ]
        
        questions = [
            "What are neural networks made of?",
            "How do you train a neural network?",
            "What happens at each node in a neural network?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer = llm.generate(question, context)
            print(f"Answer: {answer[:200]}...")  # Print first 200 chars
        
        print("\n✓ Successfully answered all questions")
        return True
        
    except LLMError as e:
        print(f"✗ Failed during Q&A: {e}")
        return False


def test_context_formatting():
    """Test that context is properly formatted."""
    print("\n=== Testing Context Formatting ===")
    
    try:
        llm = LLMInterface()
        
        context = [
            "First chunk of information.",
            "Second chunk of information.",
            "Third chunk of information."
        ]
        
        formatted = llm._format_context(context)
        print("Formatted context:")
        print(formatted)
        
        # Verify formatting
        assert "[Context 1]" in formatted
        assert "[Context 2]" in formatted
        assert "[Context 3]" in formatted
        
        print("\n✓ Context formatting is correct")
        return True
        
    except Exception as e:
        print(f"✗ Context formatting failed: {e}")
        return False


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n=== Testing Error Handling ===")
    
    try:
        llm = LLMInterface()
        
        # Test empty question
        try:
            llm.generate("", ["context"])
            print("✗ Should have raised error for empty question")
            return False
        except LLMError:
            print("✓ Correctly raised error for empty question")
        
        # Test empty context
        try:
            llm.generate("question", [])
            print("✗ Should have raised error for empty context")
            return False
        except LLMError:
            print("✓ Correctly raised error for empty context")
        
        print("\n✓ Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Run all manual tests."""
    print("=" * 60)
    print("LLM Interface Manual Integration Tests")
    print("=" * 60)
    print("\nPrerequisites:")
    print("1. LM Studio is installed and running")
    print("2. A model is loaded in LM Studio")
    print("3. Local server is started (http://localhost:1234)")
    print("=" * 60)
    
    tests = [
        ("Connection Test", test_llm_connection),
        ("Simple Generation", test_simple_generation),
        ("Lecture Content Q&A", test_lecture_content_qa),
        ("Context Formatting", test_context_formatting),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nTroubleshooting:")
        print("- Ensure LM Studio is running")
        print("- Verify a model is loaded")
        print("- Check that the server is started")
        print("- Confirm the server URL is http://localhost:1234")


if __name__ == "__main__":
    main()
