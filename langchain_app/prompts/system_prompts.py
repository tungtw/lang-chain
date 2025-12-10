"""
System prompts for various use cases
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# RAG Prompt Template
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant that answers questions based on the provided context. 
    Only use the information in the context to answer the question. If the context doesn't contain 
    the information needed to answer the question, say "I don't have enough information to answer that question.""""),
    ("human", "Context: {context}"),
    ("human", "Question: {question}"),
    ("human", "Please provide a detailed answer based on the context above.")
])


# Conversation Prompt Template
CONVERSATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the conversation history to provide context-aware responses."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# Summarization Prompt Template
SUMMARIZATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer. Create a concise summary of the provided text."),
    ("human", "Please summarize the following text: {text}")
])


# Question Answering Prompt Template
QA_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that answers questions accurately based on the provided information."),
    ("human", "Question: {question}")
])