"""
Custom LangChain agents
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_core.tools import BaseTool
from app.tools import CalculatorTool, WebSearchTool  # We'll create these next


def create_simple_agent(tools: list, llm_model="gpt-3.5-turbo"):
    """
    Creates a simple ReAct agent with provided tools
    """
    llm = ChatOpenAI(model=llm_model, temperature=0.1)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent


class CustomAgent:
    """
    A custom agent class that can be extended with additional functionality
    """
    def __init__(self, tools: list, llm_model="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.tools = tools
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str):
        """
        Run the agent with a query
        """
        return self.agent.run(query)