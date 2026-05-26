"""
Test RAG (Retrieval-Augmented Generation) System
Interactive script to test the complete RAG pipeline.

Usage:
    cd backend
    python test_query.py

This script demonstrates:
1. Semantic retrieval of relevant document chunks
2. LLM-powered answer generation grounded in retrieved context
"""
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.rag_service import RAGService


def print_separator(char="-", length=60):
    """Print a separator line"""
    print(char * length)


def print_header(title: str):
    """Print a section header"""
    print()
    print_separator("=")
    print(f"  {title}")
    print_separator("=")


def print_sources(chunks):
    """Print source information from retrieved chunks"""
    print("\n  Sources used:")

    # Get unique sources with their chunk counts
    source_counts = {}
    for chunk in chunks:
        source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1

    for source, count in source_counts.items():
        print(f"    - {source} ({count} chunk{'s' if count > 1 else ''})")


def print_retrieved_chunks(chunks, show_preview=True):
    """Print retrieved chunks with optional text preview"""
    print("\n  Retrieved Chunks:")

    for chunk in chunks:
        print(f"\n    [{chunk.rank}] {chunk.source} (similarity: {chunk.similarity_score:.4f})")

        if show_preview:
            # Show first 150 characters of each chunk
            preview = chunk.text[:150].replace('\n', ' ')
            if len(chunk.text) > 150:
                preview += "..."
            print(f"        \"{preview}\"")


def main():
    """Run the interactive RAG test"""

    print_header("Enterprise Knowledge Assistant")
    print("\n  RAG (Retrieval-Augmented Generation) Test")
    print("  Powered by Ollama (phi3:mini + nomic-embed-text)")

    # Initialize RAG service
    print("\n  Initializing...")
    print_separator()

    try:
        rag_service = RAGService()
    except Exception as e:
        print(f"\n  Error initializing RAG service: {e}")
        print("\n  Make sure:")
        print("    1. Ollama is running")
        print("    2. Models are pulled: ollama pull phi3:mini nomic-embed-text")
        print("    3. Documents are ingested: python run_ingestion.py")
        return

    # Check if there are documents indexed
    doc_count = rag_service.vector_store.get_collection_count()
    if doc_count == 0:
        print("\n  No documents found in vector database!")
        print("  Please run: python run_ingestion.py")
        return

    print(f"\n  Ready! {doc_count} chunks indexed.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'debug' to toggle chunk preview.\n")

    show_chunks = True  # Toggle for showing chunk previews

    # Interactive query loop
    while True:
        print_separator("=")
        query = input("\n  Your question: ").strip()

        # Check for exit
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n  Goodbye!")
            break

        # Toggle debug mode
        if query.lower() == 'debug':
            show_chunks = not show_chunks
            print(f"\n  Chunk preview: {'ON' if show_chunks else 'OFF'}")
            continue

        # Skip empty queries
        if not query:
            print("  Please enter a question.")
            continue

        # Process the query through the RAG pipeline
        print_separator()
        print(f"\n  Processing: \"{query}\"")

        try:
            # =================================================================
            # STEP 1: RETRIEVAL
            # Find relevant document chunks
            # =================================================================
            print("\n  [1/2] Retrieving relevant chunks...")

            # Get answer and chunks from RAG service
            answer, chunks = rag_service.generate_answer(query, top_k=3)

            # Show retrieval results
            print_sources(chunks)

            if show_chunks:
                print_retrieved_chunks(chunks)

            # =================================================================
            # STEP 2: GENERATION
            # Generate answer using LLM with retrieved context
            # =================================================================
            print("\n  [2/2] Generating answer...")

            # =================================================================
            # DISPLAY FINAL ANSWER
            # =================================================================
            print_header("Answer")
            print()
            # Print answer with proper formatting
            for line in answer.strip().split('\n'):
                print(f"  {line}")
            print()
            print_separator()

        except Exception as e:
            print(f"\n  Error during RAG pipeline: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
