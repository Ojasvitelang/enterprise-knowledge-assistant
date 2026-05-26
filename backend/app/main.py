"""
Enterprise Knowledge Assistant - FastAPI Backend

This is the main entry point for the FastAPI application.
It sets up the API server with:
- CORS middleware for cross-origin requests
- Health check endpoint
- Chat/Query routes for RAG functionality

To run the server:
    cd backend
    uvicorn app.main:app --reload

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import chat_routes


# =============================================================================
# APPLICATION SETUP
# =============================================================================
app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="""
    RAG-based AI assistant for education/training institutes.

    ## Features
    - **Document Ingestion**: Upload and index documents
    - **Semantic Search**: Find relevant information using AI
    - **Question Answering**: Get AI-generated answers grounded in your documents

    ## How it works
    1. Documents are chunked and embedded using Ollama (nomic-embed-text)
    2. Embeddings are stored in ChromaDB for fast similarity search
    3. Questions are answered using RAG with Ollama (phi3:mini)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# =============================================================================
# MIDDLEWARE
# =============================================================================
# Add CORS middleware to allow cross-origin requests
# This is needed for the Streamlit frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enterprise Knowledge Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        Service status information
    """
    return {
        "status": "healthy",
        "service": "Enterprise Knowledge Assistant"
    }


# -----------------------------------------------------------------------------
# Include Routers
# -----------------------------------------------------------------------------
# Chat routes: /chat/query, /chat/status
app.include_router(chat_routes.router)

# Future: Upload routes
# from app.routes import upload_routes
# app.include_router(upload_routes.router)
