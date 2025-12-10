"""
Example document retriever setup
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os


def create_retriever(doc_path: str, persist_directory: str = "./vectorstore"):
    """
    Creates a document retriever using ChromaDB
    """
    # Load documents
    loader = TextLoader(doc_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function,
        persist_directory=persist_directory
    )
    
    # Create retriever
    retriever = vector_store.as_retriever()
    
    return retriever