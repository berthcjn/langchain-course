import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

os.environ["LANGCHAIN_PROJECT"] = "langchain-test-langsmith-ollama"

model_name = "llama3.2:3b-instruct-fp16"
# model_name = "llama3.2:1b-instruct-fp16"

# initialize one LLM with temperature 0.0, this makes the LLM more deterministic
llm = ChatOllama(temperature=0.0, model=model_name)

system_prompt = """
Be a helpful assistant and answer the user's question.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{query}"),
])

pipeline = prompt_template | llm

query = (
    # "I have a few questions, what is the date and time right now?"
    "What is nine plus 10, minus 4 * 2, to the power of 3"
)
result = pipeline.invoke({"query": query}).content
print(result)
