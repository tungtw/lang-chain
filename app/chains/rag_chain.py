"""
RAG (Retrieval-Augmented Generation) chain
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def create_rag_chain(vector_store, llm_model="gpt-3.5-turbo"):
    """
    Creates a RAG chain with a vector store retriever
    """
    llm = ChatOpenAI(model=llm_model, temperature=0.1)
    
    # Define the RAG prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the retriever from the vector store
    retriever = vector_store.as_retriever()
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain