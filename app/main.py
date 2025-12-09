'''
Docstring for app.main
'''
# main.py
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from app.config import settings

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# 1. Define the LLM
llm = ChatOpenAI(
    model=settings.llm_model_name, #"gpt-4o-mini",
    temperature=0.7,
    api_key=api_key  # optional if set in env as OPENAI_API_KEY
)

# 2. Create a prompt template
prompt = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

# 3. Create a chain using LCEL (LangChain Expression Language)
chain = prompt | llm  # ✅ This is the modern way

# 4. Run the chain
result = chain.invoke({"adjective": "funny", "topic": "programming"})
print(result.content)  # ✅ Use .content, not ["text"]
