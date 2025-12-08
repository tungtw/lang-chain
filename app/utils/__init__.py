"""
Utility functions for the LangChain application
"""
import os
from typing import List, Dict, Any
from langchain_core.documents import Document


def load_environment_vars():
    """
    Load environment variables from .env file
    """
    from dotenv import load_dotenv
    load_dotenv()


def create_data_directory_if_not_exists():
    """
    Create data directory if it doesn't exist
    """
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/documents"):
        os.makedirs("data/documents")


def format_documents(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Format documents for consistent processing
    """
    formatted_docs = []
    for doc in docs:
        formatted_doc = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        formatted_docs.append(formatted_doc)
    return formatted_docs


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for a sentence boundary within the last 200 characters of the chunk
            search_start = max(start + chunk_size - overlap, start)
            for i in range(end, search_start, -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    
    # If no sentence boundaries were found, do a simple chunking
    if not chunks:
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks


def safe_json_loads(text: str, default=None):
    """
    Safely load JSON with error handling
    """
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename
    """
    return os.path.splitext(filename)[1].lower()


def is_supported_document_type(filename: str) -> bool:
    """
    Check if the file type is supported
    """
    supported_extensions = ['.txt', '.pdf', '.doc', '.docx', '.md', '.csv']
    return get_file_extension(filename) in supported_extensions