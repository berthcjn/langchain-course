import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSerializable
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

os.environ["LANGCHAIN_PROJECT"] = "langchain-test-langsmith-ollama"


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


# Define the multiply tool
@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'."""
    return x * y


# Define the exponentiate tool
@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'."""
    return x ** y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x


prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a math assistant. "
        "Solve problems step by step by calling tools. "
        "Use the scratchpad to remember previous tool outputs."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    # ("ai", "Scratchpad: {agent_scratchpad}"),
])

# model_name = "llama3.2:3b-instruct-fp16"
model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

tools = [add, subtract, multiply, exponentiate]

# define the agent runnable
agent: RunnableSerializable = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            # "agent_scratchpad": lambda x: x.get("agent_scratchpad", "")
        }
        | prompt
        | llm.bind_tools(tools, tool_choice="any")
)

out = agent.invoke({"input": "What is 10 + 10", "chat_history": []})

print(out.content)
print(out.tool_calls)
