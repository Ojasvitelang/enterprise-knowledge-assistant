"""
Chat/Query Routes for the Enterprise Knowledge Assistant

This module implements the API endpoints for querying the knowledge base.
The main endpoint uses RAG (Retrieval-Augmented Generation) to:
1. Find relevant document chunks
2. Generate grounded answers using an LLM

Request Flow:
    Client Request → FastAPI Router → RAGService → Response

    1. Client sends POST /query with { "question": "..." }
    2. Router validates request using Pydantic schema
    3. RAGService retrieves relevant chunks from ChromaDB
    4. RAGService generates answer using Ollama LLM
    5. Router returns { "answer": "...", "sources": [...] }
"""
from fastapi import APIRouter, HTTPException, status

from app.models.schemas import QueryRequest, QueryResponse
from app.services.rag_service import RAGService


# =============================================================================
# ROUTER SETUP
# =============================================================================
# Create router with prefix "/chat" and tag for API documentation
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable (Ollama not running)"}
    }
)


# =============================================================================
# RAG SERVICE INITIALIZATION
# =============================================================================
# Initialize RAGService as a module-level singleton
# This avoids re-loading models on every request

_rag_service = None


def get_rag_service() -> RAGService:
    """
    Get or initialize the RAG service singleton.

    This pattern ensures:
    - Models are loaded only once (not per request)
    - Service is initialized lazily (on first request)

    Returns:
        RAGService instance

    Raises:
        HTTPException: If service cannot be initialized
    """
    global _rag_service

    if _rag_service is None:
        try:
            _rag_service = RAGService()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to initialize RAG service: {str(e)}. Make sure Ollama is running."
            )

    return _rag_service


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Submit a question to get an AI-generated answer based on the indexed documents."
)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge base using RAG.

    This endpoint implements the complete RAG pipeline:

    1. RETRIEVAL: Find relevant document chunks using semantic similarity
    2. GROUNDING: Build context from retrieved chunks
    3. GENERATION: Generate answer using LLM with retrieved context

    Args:
        request: QueryRequest containing:
            - question: The user's question
            - top_k: Number of chunks to retrieve (default: 3)

    Returns:
        QueryResponse containing:
            - answer: Generated answer grounded in documents
            - sources: List of source filenames used

    Raises:
        HTTPException 400: If question is empty
        HTTPException 503: If RAG service is unavailable
        HTTPException 500: For other processing errors
    """
    # -------------------------------------------------------------------------
    # STEP 1: Validate Request
    # -------------------------------------------------------------------------
    # Check that question is not empty
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    # -------------------------------------------------------------------------
    # STEP 2: Get RAG Service
    # -------------------------------------------------------------------------
    # Get the singleton RAG service instance
    rag_service = get_rag_service()

    # Check if documents are indexed
    doc_count = rag_service.vector_store.get_collection_count()
    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed. Please run the ingestion pipeline first."
        )

    # -------------------------------------------------------------------------
    # STEP 3: Process Query through RAG Pipeline
    # -------------------------------------------------------------------------
    try:
        # Call RAG service to get answer and retrieved chunks
        # This internally:
        #   1. Embeds the question
        #   2. Retrieves relevant chunks from ChromaDB
        #   3. Builds context from chunks
        #   4. Generates answer using Ollama LLM
        answer, chunks = rag_service.generate_answer(
            query=request.question.strip(),
            top_k=request.top_k
        )

        # -------------------------------------------------------------------------
        # STEP 4: Build Response
        # -------------------------------------------------------------------------
        # Extract unique source filenames from retrieved chunks
        sources = list(set(chunk.source for chunk in chunks))

        # Return structured response
        return QueryResponse(
            answer=answer.strip(),
            sources=sources
        )

    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error processing query: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get(
    "/status",
    summary="Check RAG service status",
    description="Check if the RAG service is initialized and documents are indexed."
)
async def get_service_status():
    """
    Get the status of the RAG service.

    Returns:
        Dictionary with service status information
    """
    try:
        rag_service = get_rag_service()
        doc_count = rag_service.vector_store.get_collection_count()

        return {
            "status": "ready" if doc_count > 0 else "no_documents",
            "documents_indexed": doc_count,
            "embedding_model": rag_service.embedding_service.model_name,
            "llm_model": rag_service.llm_model
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
