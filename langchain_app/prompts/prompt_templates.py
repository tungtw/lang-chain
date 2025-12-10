"""
Example prompt templates
"""
from langchain_core.prompts import ChatPromptTemplate


# Simple QA Prompt
simple_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question as accurately as possible."),
    ("human", "{question}")
])


# RAG Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based only on the following context:
{context}

If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer that question.""""),
    ("human", "{question}")
])


# Conversation Prompt with History
conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the conversation history to provide context-aware responses."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])


def get_prompt_by_name(prompt_name: str) -> ChatPromptTemplate:
    """
    Returns a prompt template by name
    """
    prompts = {
        "simple_qa": simple_qa_prompt,
        "rag": rag_prompt,
        "conversation": conversation_prompt
    }
    
    if prompt_name in prompts:
        return prompts[prompt_name]
    else:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(prompts.keys())}")