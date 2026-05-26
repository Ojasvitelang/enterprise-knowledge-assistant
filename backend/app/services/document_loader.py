"""
Document loading service
Loads .txt files from a directory and returns structured Document objects.
"""
import os
from typing import List
from app.models.schemas import Document


class DocumentLoader:
    """Load text documents from a directory"""

    def __init__(self, directory: str):
        """
        Initialize the document loader.

        Args:
            directory: Path to the folder containing documents
        """
        self.directory = directory

    def load_single_file(self, file_path: str) -> Document:
        """
        Load a single .txt file and return a Document object.

        Args:
            file_path: Full path to the .txt file

        Returns:
            Document object with filename, content, and source path
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(file_path)

        return Document(
            filename=filename,
            content=content,
            source_path=file_path
        )

    def load_all_documents(self) -> List[Document]:
        """
        Load all .txt files from the directory.

        Returns:
            List of Document objects
        """
        documents = []

        # Check if directory exists
        if not os.path.exists(self.directory):
            print(f"Directory not found: {self.directory}")
            return documents

        # Iterate through all files in the directory
        for filename in os.listdir(self.directory):
            # Only process .txt files
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory, filename)

                try:
                    doc = self.load_single_file(file_path)
                    documents.append(doc)
                    print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        return documents
