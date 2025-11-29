"""Main entry point for the RAG chatbot CLI."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Process documents and answer questions"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the knowledge base")
    ingest_parser.add_argument("file_path", help="Path to the file to ingest")
    ingest_parser.add_argument(
        "--type",
        choices=["pdf", "audio", "video"],
        help="File type (auto-detected if not specified)"
    )
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the knowledge base")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of context chunks to retrieve (default: 5)"
    )
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the knowledge base")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # TODO: Implement command handlers
    print(f"Command '{args.command}' not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
