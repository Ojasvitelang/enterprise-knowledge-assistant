"""
RAG (Retrieval-Augmented Generation) Service

This service implements the complete RAG pipeline:
1. RETRIEVAL: Find relevant document chunks using semantic similarity
2. GROUNDING: Combine retrieved chunks into context for the LLM
3. GENERATION: Use LLM to generate an answer based ONLY on the context

The key principle is "grounding" - the LLM only answers using information
from the retrieved documents, preventing hallucination.
"""
import os
import warnings
from typing import List, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

from langchain_community.llms import Ollama

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.models.schemas import RetrievedChunk, RetrievalResult


# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
# This template instructs the LLM to:
# 1. Only use information from the provided context
# 2. Admit when information is not available
# 3. Stay grounded and avoid hallucination

RAG_PROMPT_TEMPLATE = """You are an enterprise knowledge assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, say:
"I could not find that information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:"""


class RAGService:
    """
    Complete RAG (Retrieval-Augmented Generation) Service.

    This service combines:
    - Semantic search (finding relevant documents)
    - LLM generation (creating human-readable answers)

    The result is an AI assistant that answers questions using your documents.
    """

    def __init__(
        self,
        chroma_db_path: str = None,
        collection_name: str = "knowledge_base",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "phi3:mini"
    ):
        """
        Initialize the RAG service with all components.

        Args:
            chroma_db_path: Path to ChromaDB storage
            collection_name: Name of the vector collection
            embedding_model: Ollama model for embeddings (nomic-embed-text)
            llm_model: Ollama model for generation (phi3:mini)
        """
        # Set default path if not provided
        if chroma_db_path is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            chroma_db_path = os.path.join(backend_dir, "chroma_db")

        self.llm_model = llm_model

        # ---------------------------------------------------------------------
        # COMPONENT 1: Embedding Service
        # Converts text into numerical vectors for similarity search
        # ---------------------------------------------------------------------
        self.embedding_service = EmbeddingService(model_name=embedding_model)

        # ---------------------------------------------------------------------
        # COMPONENT 2: Vector Store
        # Stores document embeddings and enables semantic search
        # ---------------------------------------------------------------------
        self.vector_store = VectorStore(
            persist_directory=chroma_db_path,
            collection_name=collection_name
        )

        # ---------------------------------------------------------------------
        # COMPONENT 3: LLM (Large Language Model)
        # Generates human-readable answers from retrieved context
        # ---------------------------------------------------------------------
        self.llm = Ollama(
            model=llm_model,
            temperature=0.1  # Low temperature for factual, consistent answers
        )

        print(f"RAG Service initialized")
        print(f"  - Embedding model: {embedding_model}")
        print(f"  - LLM model: {llm_model}")
        print(f"  - Documents indexed: {self.vector_store.get_collection_count()}")

    # =========================================================================
    # RETRIEVAL METHODS
    # =========================================================================

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        RETRIEVAL STEP: Find relevant document chunks.

        This method:
        1. Converts the user's question into an embedding vector
        2. Searches the vector database for similar chunks
        3. Returns the top-k most relevant chunks

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            List of RetrievedChunk objects ranked by similarity
        """
        # Convert query text to embedding vector
        query_embedding = self.embedding_service.embed_text(query)

        # Search vector store for similar chunks
        chunks = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return chunks

    def retrieve_with_details(self, query: str, top_k: int = 3) -> RetrievalResult:
        """
        Retrieve chunks with full metadata.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with query, chunks, and search metadata
        """
        chunks = self.retrieve(query, top_k)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            total_chunks_searched=self.vector_store.get_collection_count()
        )

    # =========================================================================
    # GROUNDING METHODS
    # =========================================================================

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        GROUNDING STEP: Combine retrieved chunks into context.

        This method takes the retrieved chunks and formats them into
        a context string that will be provided to the LLM. Each chunk
        is labeled with its source for transparency.

        Args:
            chunks: List of retrieved chunks

        Returns:
            Formatted context string
        """
        context_parts = []

        for chunk in chunks:
            # Include source information for attribution
            context_parts.append(
                f"[Source: {chunk.source}]\n{chunk.text}"
            )

        # Join chunks with clear separators
        return "\n\n---\n\n".join(context_parts)

    def build_prompt(self, context: str, question: str) -> str:
        """
        PROMPT CONSTRUCTION: Create the final prompt for the LLM.

        This method combines:
        - System instructions (answer only from context)
        - Retrieved context (the grounding information)
        - User's question

        Args:
            context: The grounding context from retrieved chunks
            question: The user's question

        Returns:
            Complete prompt string for the LLM
        """
        return RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

    # =========================================================================
    # GENERATION METHODS
    # =========================================================================

    def generate_answer(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        COMPLETE RAG PIPELINE: Retrieve + Ground + Generate

        This is the main method that implements the full RAG flow:

        1. RETRIEVAL: Find relevant chunks from the knowledge base
        2. GROUNDING: Build context from retrieved chunks
        3. PROMPT CONSTRUCTION: Create prompt with context + question
        4. GENERATION: Use LLM to generate grounded answer

        Args:
            query: User's question
            top_k: Number of chunks to use for context

        Returns:
            Tuple of (generated_answer, retrieved_chunks)
        """
        # -----------------------------------------------------------------
        # Step 1: RETRIEVAL
        # Find the most relevant document chunks for this question
        # -----------------------------------------------------------------
        chunks = self.retrieve(query, top_k=top_k)

        # Handle case where no relevant documents are found
        if not chunks:
            return (
                "I could not find any relevant information in the knowledge base.",
                []
            )

        # -----------------------------------------------------------------
        # Step 2: GROUNDING
        # Combine retrieved chunks into a context string
        # This "grounds" the LLM's response in actual documents
        # -----------------------------------------------------------------
        context = self.build_context(chunks)

        # -----------------------------------------------------------------
        # Step 3: PROMPT CONSTRUCTION
        # Build the complete prompt with instructions, context, and question
        # -----------------------------------------------------------------
        prompt = self.build_prompt(context=context, question=query)

        # -----------------------------------------------------------------
        # Step 4: GENERATION
        # Send prompt to LLM and get the answer
        # The LLM generates a response based ONLY on the provided context
        # -----------------------------------------------------------------
        answer = self.llm.invoke(prompt)

        return answer, chunks

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Simple query interface for the RAG system.

        Args:
            question: User's question
            top_k: Number of context chunks to use

        Returns:
            Dictionary with answer, sources, and chunks
        """
        answer, chunks = self.generate_answer(question, top_k=top_k)

        # Extract unique sources
        sources = list(set(chunk.source for chunk in chunks))

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks": chunks
        }
