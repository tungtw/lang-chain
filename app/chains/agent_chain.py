"""
Agent chain implementation
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from app.prompts.system_prompts import CONVERSATION_PROMPT_TEMPLATE


def create_agent_chain(tools, llm_model="gpt-3.5-turbo"):
    """
    Creates an agent chain with specified tools
    """
    llm = ChatOpenAI(model=llm_model, temperature=0.1)
    
    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_chain


def create_conversation_agent_chain(tools, llm_model="gpt-3.5-turbo"):
    """
    Creates a conversation agent chain
    """
    llm = ChatOpenAI(model=llm_model, temperature=0.1)
    
    # This is a simplified example - a full implementation would be more complex
    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_chain