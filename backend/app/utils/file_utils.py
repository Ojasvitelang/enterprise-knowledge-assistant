"""
File utility functions
"""
import os
from typing import List


def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()


def is_supported_format(filename: str) -> bool:
    """Check if file format is supported"""
    supported = [".pdf", ".docx", ".txt", ".md"]
    return get_file_extension(filename) in supported


def list_files_in_directory(directory: str) -> List[str]:
    """List all files in a directory"""
    # TODO: Implement file listing
    pass
