"""
Application configuration settings
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    app_name: str = "Enterprise Knowledge Assistant"
    debug: bool = True

    # ChromaDB Settings
    chroma_persist_directory: str = "./vector_db"
    collection_name: str = "knowledge_base"

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # LLM Settings
    # TODO: Add LLM configuration

    class Config:
        env_file = ".env"


settings = Settings()
