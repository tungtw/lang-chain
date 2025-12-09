from fastapi import FastAPI, HTTPException
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import settings

# Initialize LLM (e.g., gpt-4o-mini)
llm = ChatOpenAI(
    model=settings.llm_model_name,  # "gpt-4o-mini",
    temperature=0.7,
    api_key=settings.openai_api_key,  # optional if set in env as OPENAI_API_KEY
)

# Initialize FastAPI apiapp
apiapp = FastAPI(
    title="Local AI Chatbot",
    description="A 100% offline chatbot using Llama 3 + LangChain + FastAPI",
    version="0.1.0",
)

# Initialize LLM (runs locally via Ollama)
llm = ChatOpenAI(
    model=settings.llm_model_name,  # "gpt-4o-mini",
    temperature=0.7,
    api_key=settings.openai_api_key,  # optional if set in env as OPENAI_API_KEY
)

# Define prompt with history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, friendly AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Base chain
chain = prompt | llm

# In-memory store for chat histories (use Redis/DB in production)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Wrap with memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# Request/Response Models
class ChatRequest(BaseModel):
    input: str
    session_id: str  # e.g., user ID or conversation ID


class ChatResponse(BaseModel):
    response: str


# FastAPI Endpoint
@apiapp.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the AI assistant.

    - `input`: Your message
    - `session_id`: Unique ID to track conversation (e.g., "user_123")
    """
    try:
        config = {"configurable": {"session_id": request.session_id}}
        response = chain_with_history.invoke({"input": request.input}, config=config)
        return ChatResponse(response=response.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}") from e


# Health check
@apiapp.get("/health")
async def health_check():
    return {"status": "online", "model": "gpt-4o-mini"}
