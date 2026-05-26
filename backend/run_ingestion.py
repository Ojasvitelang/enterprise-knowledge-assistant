"""
Document Ingestion Pipeline
Loads documents, chunks them, generates embeddings, and stores in ChromaDB.

Usage:
    cd backend
    python run_ingestion.py

Requirements:
    - Ollama must be running locally
    - nomic-embed-text model must be pulled: ollama pull nomic-embed-text
"""
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_loader import DocumentLoader
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


def main():
    """Run the complete document ingestion pipeline"""

    print("=" * 60)
    print("Document Ingestion Pipeline")
    print("=" * 60)

    # Configuration
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    sample_docs_path = os.path.abspath(os.path.join(backend_dir, "..", "sample_documents"))
    chroma_db_path = os.path.join(backend_dir, "chroma_db")

    print(f"\nSource folder: {sample_docs_path}")
    print(f"Vector DB path: {chroma_db_path}")

    # =========================================================
    # Step 1: Load Documents
    # =========================================================
    print("\n" + "-" * 60)
    print("Step 1: Loading Documents")
    print("-" * 60)

    loader = DocumentLoader(directory=sample_docs_path)
    documents = loader.load_all_documents()

    print(f"\nTotal documents loaded: {len(documents)}")

    if len(documents) == 0:
        print("\nNo .txt files found. Add files to sample_documents/ and try again.")
        return

    # =========================================================
    # Step 2: Chunk Documents
    # =========================================================
    print("\n" + "-" * 60)
    print("Step 2: Chunking Documents")
    print("-" * 60)
    print("Settings: chunk_size=500, chunk_overlap=50\n")

    chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)

    print(f"\nTotal chunks created: {len(chunks)}")

    # =========================================================
    # Step 3: Generate Embeddings
    # =========================================================
    print("\n" + "-" * 60)
    print("Step 3: Generating Embeddings")
    print("-" * 60)

    embedding_service = EmbeddingService(model_name="nomic-embed-text")

    # Extract texts from chunks
    chunk_texts = [chunk.text for chunk in chunks]

    print(f"\nGenerating embeddings for {len(chunk_texts)} chunks...")
    embeddings = embedding_service.embed_texts(chunk_texts)

    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # =========================================================
    # Step 4: Store in ChromaDB
    # =========================================================
    print("\n" + "-" * 60)
    print("Step 4: Storing in ChromaDB")
    print("-" * 60)

    vector_store = VectorStore(
        persist_directory=chroma_db_path,
        collection_name="knowledge_base"
    )

    # Clear existing data for fresh ingestion
    print("\nClearing existing data...")
    vector_store.clear_collection()

    # Prepare metadata and IDs
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        metadatas.append({
            "source": chunk.source_document,
            "chunk_index": chunk.chunk_index
        })
        ids.append(f"{chunk.source_document}_{chunk.chunk_index}")

    # Add to vector store
    print("Adding vectors to database...")
    vectors_added = vector_store.add_documents(
        texts=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"\nTotal documents loaded:  {len(documents)}")
    print(f"Total chunks created:    {len(chunks)}")
    print(f"Total vectors stored:    {vector_store.get_collection_count()}")

    # Preview stored data
    print("\n" + "-" * 60)
    print("Stored Documents Preview:")
    print("-" * 60)
    for doc in documents:
        doc_chunks = [c for c in chunks if c.source_document == doc.filename]
        print(f"  {doc.filename}: {len(doc_chunks)} chunks")

    print("\n" + "=" * 60)
    print("Vector database ready for queries!")
    print("=" * 60)


if __name__ == "__main__":
    main()
