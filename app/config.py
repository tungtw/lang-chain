"""
Configuration management using Pydantic Settings.
Loads environment variables and defines application defaults.
"""
# pylint: disable=too-few-public-methods
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # LLM Configuration
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"

    # Vector Store Configuration
    vector_store_path: str = "./vectorstore"
    embeddings_model: str = "all-MiniLM-L6-v2"

    # Application Configuration
    debug: bool = False
    app_name: str = "LangChain Application"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

settings = Settings()
