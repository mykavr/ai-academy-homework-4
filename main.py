"""Main entry point for the RAG chatbot CLI."""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional

from src.rag.chatbot import RAGChatbot
from src.config import default_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_file_type(file_path: str) -> Optional[str]:
    """
    Auto-detect file type based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type ('pdf', 'audio', 'video') or None if unknown
    """
    ext = Path(file_path).suffix.lower()
    
    # PDF extensions
    if ext == '.pdf':
        return 'pdf'
    
    # Audio extensions
    audio_exts = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
    if ext in audio_exts:
        return 'audio'
    
    # Video extensions
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    if ext in video_exts:
        return 'video'
    
    return None


def handle_ingest(chatbot: RAGChatbot, file_path: str, file_type: Optional[str] = None):
    """
    Handle document ingestion command.
    
    Args:
        chatbot: RAGChatbot instance
        file_path: Path to the file to ingest
        file_type: Explicit file type or None for auto-detection
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    # Auto-detect file type if not specified
    if file_type is None:
        file_type = detect_file_type(file_path)
        if file_type is None:
            print(f"❌ Error: Could not detect file type for: {file_path}")
            print("Please specify the file type using --type")
            sys.exit(1)
        print(f"📄 Detected file type: {file_type}")
    
    print(f"📥 Ingesting {file_type} file: {file_path}")
    print("⏳ Processing... (this may take a while)")
    
    try:
        # Call appropriate ingestion method
        if file_type == 'pdf':
            result = chatbot.ingest_pdf(file_path)
        elif file_type == 'audio':
            result = chatbot.ingest_audio(file_path)
        elif file_type == 'video':
            result = chatbot.ingest_video(file_path)
        else:
            print(f"❌ Error: Unsupported file type: {file_type}")
            sys.exit(1)
        
        # Display results
        if result['success']:
            print(f"✅ Successfully ingested: {result['source']}")
            print(f"📊 Added {result['chunks_added']} chunks to knowledge base")
        else:
            print(f"⚠️  Warning: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        logger.exception("Ingestion failed")
        sys.exit(1)


def handle_ask(chatbot: RAGChatbot, question: str, top_k: int, show_context: bool = True):
    """
    Handle question answering command.
    
    Args:
        chatbot: RAGChatbot instance
        question: User's question
        top_k: Number of context chunks to retrieve
        show_context: Whether to display retrieved context
    """
    print(f"❓ Question: {question}")
    print("⏳ Searching knowledge base and generating answer...")
    print()
    
    try:
        result = chatbot.ask(question, top_k=top_k)
        
        # Display answer
        print("=" * 80)
        print("💡 ANSWER:")
        print("=" * 80)
        print(result['answer'])
        print()
        
        # Display sources
        if result.get('sources'):
            print("=" * 80)
            print("📚 SOURCES:")
            print("=" * 80)
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source}")
            print()
        
        # Display context if requested
        if show_context and result.get('context'):
            print("=" * 80)
            print(f"📖 RETRIEVED CONTEXT ({len(result['context'])} chunks):")
            print("=" * 80)
            for i, context in enumerate(result['context'], 1):
                print(f"\n--- Chunk {i} ---")
                # Truncate long context for display
                if len(context) > 500:
                    print(context[:500] + "...")
                else:
                    print(context)
            print()
        
        # Check for errors
        if result.get('error'):
            print(f"⚠️  Note: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error answering question: {str(e)}")
        logger.exception("Question answering failed")
        sys.exit(1)


def handle_interactive(chatbot: RAGChatbot, top_k: int):
    """
    Handle interactive question-answering mode.
    
    Args:
        chatbot: RAGChatbot instance
        top_k: Number of context chunks to retrieve
    """
    print("=" * 80)
    print("🤖 RAG CHATBOT - Interactive Mode")
    print("=" * 80)
    print("Ask questions about your knowledge base.")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'stats' to see knowledge base statistics")
    print("  - Type 'quit' or 'exit' to leave")
    print("=" * 80)
    print()
    
    while True:
        try:
            # Get user input
            question = input("You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Check for stats command
            if question.lower() == 'stats':
                stats = chatbot.get_stats()
                print(f"📊 Knowledge Base Statistics:")
                print(f"   Total chunks: {stats['total_documents']}")
                print(f"   Collection: {stats['collection_name']}")
                print()
                continue
            
            # Skip empty questions
            if not question:
                continue
            
            print()
            
            # Answer the question
            result = chatbot.ask(question, top_k=top_k)
            
            # Display answer
            print("Bot:", result['answer'])
            print()
            
            # Display sources
            if result.get('sources'):
                print(f"📚 Sources: {', '.join(result['sources'])}")
                print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            logger.exception("Error in interactive mode")


def handle_clear(chatbot: RAGChatbot):
    """
    Handle knowledge base clearing command.
    
    Args:
        chatbot: RAGChatbot instance
    """
    # Ask for confirmation
    response = input("⚠️  Are you sure you want to clear the knowledge base? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Clear operation cancelled")
        return
    
    try:
        chatbot.clear_knowledge_base()
        print("✅ Knowledge base cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing knowledge base: {str(e)}")
        logger.exception("Clear operation failed")
        sys.exit(1)


def handle_stats(chatbot: RAGChatbot):
    """
    Handle statistics display command.
    
    Args:
        chatbot: RAGChatbot instance
    """
    try:
        stats = chatbot.get_stats()
        print("=" * 80)
        print("📊 KNOWLEDGE BASE STATISTICS")
        print("=" * 80)
        print(f"Total chunks: {stats['total_documents']}")
        print(f"Collection: {stats['collection_name']}")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Error getting statistics: {str(e)}")
        logger.exception("Stats retrieval failed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Process documents and answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a PDF file
  python main.py ingest document.pdf
  
  # Ingest an audio file
  python main.py ingest lecture.wav --type audio
  
  # Ask a question
  python main.py ask "What is the main topic?"
  
  # Interactive mode
  python main.py interactive
  
  # View statistics
  python main.py stats
  
  # Clear knowledge base
  python main.py clear
        """
    )
    
    # Global options
    parser.add_argument(
        "--db-path",
        default="./qdrant_storage",
        help="Path to vector database storage (default: ./qdrant_storage)"
    )
    parser.add_argument(
        "--model-path",
        default=default_config.vosk_model_path,
        help=f"Path to Vosk model (default: {default_config.vosk_model_path})"
    )
    parser.add_argument(
        "--lm-studio-url",
        default=default_config.lm_studio_url,
        help=f"LM Studio server URL (default: {default_config.lm_studio_url})"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into the knowledge base"
    )
    ingest_parser.add_argument("file_path", help="Path to the file to ingest")
    ingest_parser.add_argument(
        "--type",
        choices=["pdf", "audio", "video"],
        help="File type (auto-detected if not specified)"
    )
    
    # Ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question about the knowledge base"
    )
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=default_config.top_k,
        help=f"Number of context chunks to retrieve (default: {default_config.top_k})"
    )
    ask_parser.add_argument(
        "--no-context",
        action="store_true",
        help="Don't display retrieved context chunks"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive question-answering mode"
    )
    interactive_parser.add_argument(
        "--top-k",
        type=int,
        default=default_config.top_k,
        help=f"Number of context chunks to retrieve (default: {default_config.top_k})"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Display knowledge base statistics"
    )
    
    # Clear command
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear the knowledge base"
    )
    
    args = parser.parse_args()
    
    # Show help if no command specified
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Initialize chatbot
    print("🚀 Initializing RAG Chatbot...")
    try:
        chatbot = RAGChatbot(
            model_path=args.model_path,
            persist_directory=args.db_path,
            lm_studio_url=args.lm_studio_url,
            llm_timeout=default_config.llm_timeout,
            llm_debug_logging=default_config.llm_debug_logging
        )
        print("✅ Chatbot initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {str(e)}")
        logger.exception("Initialization failed")
        sys.exit(1)
    
    # Handle commands
    try:
        if args.command == "ingest":
            handle_ingest(chatbot, args.file_path, args.type)
        
        elif args.command == "ask":
            handle_ask(chatbot, args.question, args.top_k, not args.no_context)
        
        elif args.command == "interactive":
            handle_interactive(chatbot, args.top_k)
        
        elif args.command == "stats":
            handle_stats(chatbot)
        
        elif args.command == "clear":
            handle_clear(chatbot)
        
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
    
    finally:
        # Clean up
        chatbot.close()


if __name__ == "__main__":
    main()
