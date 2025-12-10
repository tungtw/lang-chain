"""
Custom Pydantic models for the LangChain application
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class DocumentModel(BaseModel):
    """
    Model for document representation
    """
    id: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = {}
    source: Optional[str] = None


class QueryModel(BaseModel):
    """
    Model for query input
    """
    query: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ResponseModel(BaseModel):
    """
    Model for response output
    """
    response: str
    source_documents: List[str] = []
    metadata: Dict[str, Any] = {}


class AgentConfig(BaseModel):
    """
    Model for agent configuration
    """
    max_iterations: int = 10
    early_stopping: bool = True
    return_intermediate_steps: bool = False