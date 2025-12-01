"""
LLM Interface for connecting to LM Studio local server.

This module provides an interface to interact with LM Studio's OpenAI-compatible API
for generating answers based on retrieved context chunks.
"""

from typing import List
from openai import OpenAI
import httpx


class LLMError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class LLMInterface:
    """
    Interface for connecting to LM Studio local server and generating responses.
    
    LM Studio provides an OpenAI-compatible API that runs locally without requiring
    API keys or cloud connections.
    """
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "local-model"):
        """
        Initialize the LLM interface.
        
        Args:
            base_url: The base URL for the LM Studio server (default: http://localhost:1234/v1)
            model: The model identifier (default: "local-model")
        """
        self.base_url = base_url
        self.model = model
        
        try:
            # Create a custom httpx client to work around Python 3.14 compatibility issues
            http_client = httpx.Client(
                base_url=base_url,
                timeout=30.0
            )
            
            # Initialize OpenAI client pointing to LM Studio
            self.client = OpenAI(
                base_url=base_url,
                api_key="not-needed",  # LM Studio doesn't require an API key
                http_client=http_client
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM client: {str(e)}")
    
    def generate(self, question: str, context: List[str]) -> str:
        """
        Generate an answer to a question using retrieved context chunks.
        
        Args:
            question: The user's question
            context: List of retrieved text chunks to use as context
            
        Returns:
            The generated answer as a string
            
        Raises:
            LLMError: If answer generation fails
        """
        if not question or not question.strip():
            raise LLMError("Question cannot be empty")
        
        if not context:
            raise LLMError("Context cannot be empty")
        
        try:
            # Format the context chunks with clear separation
            formatted_context = self._format_context(context)
            
            # Create the prompt
            prompt = self._create_prompt(question, formatted_context)
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant answering questions about lecture content. "
                                   "Answer based on the context provided. If the context doesn't contain "
                                   "relevant information, say so clearly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract the answer
            answer = response.choices[0].message.content
            
            if not answer or not answer.strip():
                raise LLMError("LLM returned an empty response")
            
            return answer.strip()
            
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            
            # Provide actionable error message
            error_msg = f"Failed to generate answer: {str(e)}"
            
            if "Connection" in str(e) or "connect" in str(e).lower():
                error_msg += "\n\nTroubleshooting:\n"
                error_msg += "1. Ensure LM Studio is running\n"
                error_msg += "2. Verify the server is started in LM Studio\n"
                error_msg += f"3. Check that the server URL is correct: {self.base_url}\n"
                error_msg += "4. Ensure a model is loaded in LM Studio"
            
            raise LLMError(error_msg)
    
    def _format_context(self, context: List[str]) -> str:
        """
        Format context chunks with clear separation and numbering.
        
        Args:
            context: List of text chunks
            
        Returns:
            Formatted context string
        """
        formatted_chunks = []
        for i, chunk in enumerate(context, 1):
            formatted_chunks.append(f"[Context {i}]\n{chunk}")
        
        return "\n\n".join(formatted_chunks)
    
    def _create_prompt(self, question: str, formatted_context: str) -> str:
        """
        Create the prompt for the LLM.
        
        Args:
            question: The user's question
            formatted_context: The formatted context chunks
            
        Returns:
            The complete prompt string
        """
        prompt = f"""Context:
{formatted_context}

Question: {question}

Answer based on the context provided above."""
        
        return prompt
