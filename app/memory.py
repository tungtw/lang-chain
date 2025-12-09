"""
Build a stateful chatbot using local LLM (Ollama) that:

Remembers your name across turns
Uses in-memory chat history
Is built with LCEL (LangChain Expression Language) — the modern way

Old Way (v0.1)	        New Way (v0.2+)
`ConversationChain`	    `RunnableWithMessageHistory`
Deprecated	            **Recommended**

"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from app.config import settings

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize LLM (e.g., gpt-4o-mini)
llm = ChatOpenAI(
    model=settings.llm_model_name,  # "gpt-4o-mini",
    temperature=0.7,
    api_key=api_key,  # optional if set in env as OPENAI_API_KEY
)

# Define prompt with chat history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
# Create chain
chain = prompt | llm

# Store chat history in memory (simple in-memory)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Wrap chain with memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
# Test conversation
config = {"configurable": {"session_id": "abc123"}}
response = chain_with_history.invoke({"input": "My name is Alex"}, config=config)
print("Bot:", response.content)

response = chain_with_history.invoke({"input": "What's my name?"}, config=config)
print("Bot:", response.content)
