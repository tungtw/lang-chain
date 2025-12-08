"""
Tests for utility functions
"""
import pytest
import os
from app.utils import chunk_text, is_supported_document_type, get_file_extension


def test_chunk_text():
    """Test text chunking function"""
    text = "This is a sample text. " * 100  # Create a long text
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_get_file_extension():
    """Test file extension extraction"""
    assert get_file_extension("document.pdf") == ".pdf"
    assert get_file_extension("image.PNG") == ".png"  # Should be case insensitive
    assert get_file_extension("archive.tar.gz") == ".gz"


def test_is_supported_document_type():
    """Test document type checking"""
    assert is_supported_document_type("document.pdf") is True
    assert is_supported_document_type("file.txt") is True
    assert is_supported_document_type("data.csv") is True
    assert is_supported_document_type("image.jpg") is False


def test_placeholder():
    """Placeholder test to ensure test structure works"""
    assert True