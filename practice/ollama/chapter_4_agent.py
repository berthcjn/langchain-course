import os
from datetime import datetime

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

os.environ["LANGCHAIN_PROJECT"] = "langchain-test-langsmith-ollama"


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'. Both 'x' and 'y' are numbers."""
    return x + y


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' and 'y'. Both 'x' and 'y' are numbers."""
    return x * y


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the power of 'y'. Both 'x' and 'y' are numbers."""
    return x ** y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'. Both 'x' and 'y' are numbers."""
    return y - x


@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. Using the tools provided you must answer the "
        "user's questions. To use the tool you must always provide the correct JSON "
        "format for the tool input. For example, if adding two numbers you would use "
        "the `add` tool, passing the two numbers to the `x` and `y` parameters."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

model_name = "llama3.2:3b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

memory = ConversationBufferMemory(
    memory_key="chat_history",  # must align with MessagesPlaceholder variable_name
    return_messages=True  # to return Message objects
)

tools = [add, subtract, multiply, exponentiate, get_current_datetime]

agent = create_tool_calling_agent(
    llm=llm, tools=tools, prompt=prompt
)

# result = agent.invoke({
#     "input": "what is 10 multiplied by 7?",
#     "chat_history": memory.chat_memory.messages,
#     "intermediate_steps": []  # agent will append it's internal steps here
# })

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# res_1 = agent_executor.invoke({
#     "input": "what is 10 exponentiated by 7?",
#     "chat_history": memory.chat_memory.messages,
# })
#
# print(res_1['output'])

# res_2 = agent_executor.invoke({
#     "input": "My name is Josh",
#     "chat_history": memory
# })

# print(res_2['output'])
#
res_3 = agent_executor.invoke({
    "input": "I have a few questions, what is the date and time right now?",
    "chat_history": memory
})

print(res_3['output'])

# res_3 = agent_executor.invoke({
#     "input": "What is nine added by 10, minus 4 * 2, to the power of 3",
#     "chat_history": memory
# })
#
# print(res_3['output'])
