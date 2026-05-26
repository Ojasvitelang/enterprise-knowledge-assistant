"""
Embedding service
Generates vector embeddings using Ollama with nomic-embed-text model.
"""
# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore")

from typing import List
from langchain_community.embeddings import OllamaEmbeddings


class EmbeddingService:
    """Generate embeddings using Ollama"""

    def __init__(self, model_name: str = "nomic-embed-text"):
        """
        Initialize the embedding service with Ollama.

        Args:
            model_name: Name of the Ollama embedding model (default: nomic-embed-text)
        """
        self.model_name = model_name

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.model_name
        )

        print(f"Embedding model initialized: {self.model_name}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
