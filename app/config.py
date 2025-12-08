from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_model_name: str = "gpt-3.5-turbo"
    
    # Vector Store Configuration
    vector_store_path: str = "./vectorstore"
    embeddings_model: str = "all-MiniLM-L6-v2"
    
    # Application Configuration
    debug: bool = False
    app_name: str = "LangChain Application"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create a single instance of settings
settings = Settings()