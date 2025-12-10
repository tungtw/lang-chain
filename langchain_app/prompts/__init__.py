"""
Directory for prompt templates used in the application
"""
from .prompt_templates import get_prompt_by_name, simple_qa_prompt, rag_prompt, conversation_prompt
from .system_prompts import (
    RAG_PROMPT_TEMPLATE,
    CONVERSATION_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE
)

__all__ = [
    "get_prompt_by_name",
    "simple_qa_prompt",
    "rag_prompt",
    "conversation_prompt",
    "RAG_PROMPT_TEMPLATE",
    "CONVERSATION_PROMPT_TEMPLATE",
    "SUMMARIZATION_PROMPT_TEMPLATE",
    "QA_PROMPT_TEMPLATE"
]