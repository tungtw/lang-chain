# LangChain Application

This is a LangChain-based application for building LLM-powered tools and services.

## Project Structure

```
my-langchain-app/
│
├── app/                          # Main application source code
│   ├── __init__.py
│   ├── main.py                   # Entry point (FastAPI, Streamlit, CLI, etc.)
│   ├── config.py                 # Configuration (API keys, defaults, etc.)
│   ├── models/                   # Custom Pydantic models or data schemas
│   │   └── __init__.py
│   ├── chains/                   # LangChain chains (custom or modularized)
│   │   ├── __init__.py
│   │   ├── rag_chain.py
│   │   └── agent_chain.py
│   ├── agents/                   # Custom agents (if using LangChain agents)
│   │   └── __init__.py
│   ├── tools/                    # Custom LangChain tools
│   │   └── __init__.py
│   ├── prompts/                  # Template files or prompt engineering
│   │   ├── __init__.py
│   │   ├── system_prompts.py
│   │   └── templates/
│   ├── retrievers/               # Custom retrievers or vector store logic
│   │   └── __init__.py
│   └── utils/                    # Helper functions (e.g., document loading, chunking)
│       └── __init__.py
│
├── data/                         # Raw or processed data (PDFs, TXT, etc.)
│   └── documents/
│
├── vectorstore/                  # Persistent vector database (e.g., FAISS, Chroma)
│   └── .gitignore                # Usually ignored in version control
│
├── tests/                        # Unit and integration tests
│   ├── __init__.py
│   ├── test_chains.py
│   └── test_utils.py
│
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API keys, etc.)
├── .gitignore
├── README.md
└── pyproject.toml or setup.py    # Optional for packaging
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run the application:
   ```bash
   cd app
   python main.py
   ```

## Features

- Configurable LLM settings
- Modular chain architecture (RAG, agents, etc.)
- Custom tools and agents
- Prompt management with system prompts
- Document retrieval capabilities
- FastAPI web interface
- Testing framework
- Utility functions for document processing
- Pydantic models for data validation