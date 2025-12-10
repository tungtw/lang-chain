*** Think of it as Lego blocks for AI apps:
- 🧩 Models: GPT, Llama, Claude, etc.
- 🧩 Prompts: Templates for consistent input
- 🧩 Chains: Sequences of steps (e.g., retrieve → generate)
- 🧩 Agents: LLMs that use tools (search, calculate, etc.)
- 🧩 Memory: Remember past interactions
- 🧩 Indexes: Load & retrieve your documents (for RAG)

*** 
```
This is the core pattern in LangChain: compose reusable blocks.
ChatOpenAI     | A ChatModel (message-based LLM wrapper)           | Abstracts API calls; switch models easily
PromptTemplate | A reusable prompt with placeholders ({adjective}) | Ensures consistent, structured input
LLMChain       | A chain = prompt + LLM                            | Composes components into a workflow
```
🧩 Key LangChain Concepts (Simplified)
- 1.    Models
        LLM: Simple text-in, text-out (OpenAI())
        ChatModel: Message-based (ChatOpenAI()) → use this for modern LLMs
2.      Prompts
        PromptTemplate: For simple strings
        ChatPromptTemplate: For chat-style (system/user/assistant messages)
3. Chains
        Prebuilt: LLMChain, RetrievalQA, ConversationChain
        Custom: Combine any components
4. Components Are Swappable
        Swap ChatOpenAI → ChatOllama (for local LLMs)
        Swap PromptTemplate → custom logic
        No vendor lock-in!

Next:
Prompt Engineering
Memory: Make it remember past jokes
RAG: Answer questions about your notes
Agents: Let the LLM search the web for fresh jokes