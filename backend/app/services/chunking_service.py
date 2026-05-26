"""
Text chunking service
Splits documents into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models.schemas import Document, TextChunk


class ChunkingService:
    """Split documents into smaller chunks for embedding"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Maximum size of each chunk (default: 500 characters)
            chunk_overlap: Number of overlapping characters between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """
        Split a single document into chunks.

        Args:
            document: Document object to split

        Returns:
            List of TextChunk objects
        """
        # Split the text into chunks
        text_chunks = self.text_splitter.split_text(document.content)

        # Create TextChunk objects with metadata
        chunks = []
        for index, chunk_text in enumerate(text_chunks):
            chunk = TextChunk(
                text=chunk_text,
                chunk_index=index,
                source_document=document.filename
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of all TextChunk objects from all documents
        """
        all_chunks = []

        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
            print(f"  {document.filename}: {len(chunks)} chunks")

        return all_chunks
