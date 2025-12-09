Fantastic! 🎉 Let’s begin your LangChain journey with **Topic #1: LangChain Fundamentals**.

We’ll keep it practical: **no fluff, just hands-on code** that runs and teaches you the core ideas.

---

## 🧱 What Is LangChain? (The 60-Second Version)

> **LangChain is not an AI model** — it’s a **Python framework** that helps you **connect LLMs to your data, logic, and tools** to build intelligent applications.

Think of it as **Lego blocks for AI apps**:
- 🧩 **Models**: GPT, Llama, Claude, etc.
- 🧩 **Prompts**: Templates for consistent input
- 🧩 **Chains**: Sequences of steps (e.g., retrieve → generate)
- 🧩 **Agents**: LLMs that *use tools* (search, calculate, etc.)
- 🧩 **Memory**: Remember past interactions
- 🧩 **Indexes**: Load & retrieve your documents (for RAG)

---

## 🔧 Step 1: Install LangChain (Minimal Setup)

We’ll start with **OpenAI** (you can switch to free/local models later).

```bash
# Create a virtual environment (recommended)
python -m venv langchain-env
source langchain-env/bin/activate  # Linux/Mac
# langchain-env\Scripts\activate  # Windows

# Install core packages
pip install langchain langchain-openai python-dotenv
```

> 💡 **Note**: You’ll need an [OpenAI API key](https://platform.openai.com/api-keys).  
> Save it in a `.env` file:
> ```
> OPENAI_API_KEY=your-api-key-here
> ```

---

## 🧪 Step 2: Your First LangChain Program

### Goal: Use an LLM to generate a joke using a **prompt template**.

```python
# main.py
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 1. Define the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # cheaper/faster than gpt-4o
    temperature=0.7,      # creativity (0 = deterministic, 1 = random)
    api_key=api_key
)

# 2. Create a prompt template
prompt = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}."
)

# 3. Create a chain (prompt + LLM)
chain = LLMChain(llm=llm, prompt=prompt)

# 4. Run the chain
result = chain.invoke({"adjective": "funny", "topic": "programming"})
print(result["text"])
```

### ✅ Run it:
```bash
python main.py
```

You’ll get something like:
> Why do programmers prefer dark mode?  
> Because light attracts bugs! 😄

---

## 🔍 What Just Happened? (The Fundamentals)

| Concept | What It Is | Why It Matters |
|--------|-----------|----------------|
| **`ChatOpenAI`** | A **ChatModel** (message-based LLM wrapper) | Abstracts API calls; switch models easily |
| **`PromptTemplate`** | A reusable prompt with **placeholders** (`{adjective}`) | Ensures consistent, structured input |
| **`LLMChain`** | A **chain** = prompt + LLM | Composes components into a workflow |

> 💡 This is the **core pattern** in LangChain: **compose reusable blocks**.

---

## 🧩 Key LangChain Concepts (Simplified)

### 1. **Models**
- **LLM**: Simple text-in, text-out (`OpenAI()`)
- **ChatModel**: Message-based (`ChatOpenAI()`) → **use this for modern LLMs**

### 2. **Prompts**
- `PromptTemplate`: For simple strings
- `ChatPromptTemplate`: For chat-style (system/user/assistant messages)

### 3. **Chains**
- **Prebuilt**: `LLMChain`, `RetrievalQA`, `ConversationChain`
- **Custom**: Combine any components

### 4. **Components Are Swappable**
- Swap `ChatOpenAI` → `ChatOllama` (for local LLMs)
- Swap `PromptTemplate` → custom logic
- No vendor lock-in!

---

## 🧪 Try This: Modify the Prompt

Change the template to:
```python
prompt = PromptTemplate.from_template(
    "Explain {topic} like I'm 5 years old. Use at most 2 sentences."
)
```

Then run:
```python
result = chain.invoke({"topic": "blockchain"})
```

> 🌟 **You just built a "Explain Like I’m 5" AI!**

---

## 🧭 What’s Next?

Now that you’ve seen the **core trio** (Model + Prompt + Chain), you can explore:

- **Memory**: Make it remember past jokes
- **RAG**: Answer questions about your notes
- **Agents**: Let the LLM search the web for fresh jokes

---

## ❓ Common Beginner Questions

**Q: Do I need OpenAI? Can I use free models?**  
✅ **Yes!** Replace `ChatOpenAI` with:
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1:8b")  # Runs locally!
```
→ Install with `pip install langchain-ollama` and [Ollama](https://ollama.com)

**Q: What’s the difference between `invoke()` and `run()`?**  
- Use `.invoke()` (newer LangChain v0.1+)
- `.run()` is legacy

**Q: Where do I find more templates?**  
Check the [LangChain Prompt Hub](https://smith.langchain.com/hub)!

---

## ✅ Your Homework (5 minutes)

1. Run the joke generator
2. Change the `adjective` and `topic`
3. Try a different model (`gpt-4o`, `gpt-4o-mini`, or `llama3` if using Ollama)

---

When you’re ready, say:  
➡️ **“Next: Prompt Engineering”**  
or  
➡️ **“Let’s build a chatbot with memory”**

Your LangChain journey has officially begun! 🚀