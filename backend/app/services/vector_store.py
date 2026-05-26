"""
Vector store service
Handles ChromaDB operations for storing and retrieving embeddings.
"""
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

from app.models.schemas import RetrievedChunk


class VectorStore:
    """ChromaDB vector store for document embeddings"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "knowledge_base"
    ):
        """
        Initialize ChromaDB with local persistence.

        Args:
            persist_directory: Directory to store the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Enterprise Knowledge Base"}
        )

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> int:
        """
        Add documents with embeddings to the vector store.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts (source filename, chunk index, etc.)
            ids: List of unique IDs for each document

        Returns:
            Number of documents added
        """
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return len(texts)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query similar documents by embedding (raw ChromaDB response).

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            Dictionary with documents, distances, and metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        Search for similar chunks and return structured results.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of RetrievedChunk objects with similarity scores
        """
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to RetrievedChunk objects
        retrieved_chunks = []

        # ChromaDB returns nested lists, extract first (and only) query results
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for rank, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Convert distance to similarity score
            # ChromaDB uses L2 distance by default, lower = more similar
            # Convert to similarity: 1 / (1 + distance)
            similarity_score = 1 / (1 + distance)

            chunk = RetrievedChunk(
                text=doc,
                source=metadata.get("source", "unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                similarity_score=round(similarity_score, 4),
                rank=rank + 1
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents stored
        """
        return self.collection.count()

    def clear_collection(self):
        """
        Delete all documents from the collection.
        """
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Enterprise Knowledge Base"}
        )
        print("Collection cleared.")
