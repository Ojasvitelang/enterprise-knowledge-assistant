"""
Pydantic schemas for request/response models
"""
from pydantic import BaseModel
from typing import List, Optional


class Document(BaseModel):
    """Represents a loaded document"""
    filename: str
    content: str
    source_path: str


class TextChunk(BaseModel):
    """Represents a chunk of text from a document"""
    text: str
    chunk_index: int
    source_document: str


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from similarity search"""
    text: str
    source: str
    chunk_index: int
    similarity_score: float
    rank: int


class QueryRequest(BaseModel):
    """
    Request model for RAG queries.

    Attributes:
        question: The user's question to be answered
        top_k: Number of chunks to retrieve for context (default: 3)
    """
    question: str
    top_k: Optional[int] = 3

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the attendance policy?",
                "top_k": 3
            }
        }


class RetrievalResult(BaseModel):
    """Response model for retrieval (without LLM answer)"""
    query: str
    chunks: List[RetrievedChunk]
    total_chunks_searched: int


class QueryResponse(BaseModel):
    """
    Response model for RAG queries.

    Attributes:
        answer: The generated answer grounded in retrieved documents
        sources: List of source document filenames used to generate the answer
    """
    answer: str
    sources: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The minimum attendance required is 80%.",
                "sources": ["company_policies.txt"]
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    filename: str
    status: str
    chunks_created: int
