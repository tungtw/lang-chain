"""
Example LangChain chain - a simple Q&A chain with memory
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.memory import ConversationBufferMemory
from langchain_core.chains import LLMChain


def create_simple_qa_chain():
    """
    Creates a simple question-answering chain
    """
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}")
    ])
    
    # Create the chain: prompt -> LLM -> output parser
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def create_memory_chain():
    """
    Creates a conversational chain with memory
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Create a prompt with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's questions based on the conversation history."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Note: This is a simplified example - you'd need to connect this with actual memory
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return chain